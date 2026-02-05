"""
Semantic Router Training Script
使用 Sentence Transformers + Logistic Regression 實作語意路由器
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42):
    """設置全局隨機種子以確保可重現性"""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 盡量讓 GPU deterministic（可能變慢）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 2.x 可用（若遇到不支援的算子會丟錯）
        # torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class SemanticRouter:
    """
    語意路由器：基於 embedding 的文本分類器
    - Fast Path (Label 0): classification, summarization
    - Slow Path (Label 1): creative_writing, general_qa
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        text_format: str = 'instruction_sep_context',
        max_context_chars: int = 2000,
        C: float = 1.0
    ):
        """
        Args:
            model_name: SentenceTransformer 模型名稱
                推薦選項:
                - 'all-MiniLM-L6-v2': 輕量快速 (80MB, 384 dim)
                - 'all-mpnet-base-v2': 效果更好但較慢 (420MB, 768 dim)
                - 'paraphrase-multilingual-MiniLM-L12-v2': 支援中文
            text_format: 文本格式選項
                - 'instruction_only': 只使用 instruction
                - 'instruction_sep_context': instruction + [SEP] + context
            max_context_chars: context 的最大字符數，超過會截斷
            C: Logistic Regression 的正則化參數（C=1.0）
        """
        logger.info(f"Loading SentenceTransformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # 處理類別不平衡
        )
        self.model_name = model_name
        self.text_format = text_format
        self.max_context_chars = max_context_chars
        
    def prepare_text(self, item: dict) -> str:
        """
        將資料轉換為可用於 embedding 的文本
        
        Args:
            item: 包含 instruction 和 context 的字典
            
        Returns:
            處理後的文本字符串
        """
        instruction = item.get('instruction', '').strip()
        context = item.get('context', '').strip()
        
        # 根據 text_format 決定格式
        if self.text_format == 'instruction_only':
            # 只使用 instruction
            return instruction
        elif self.text_format == 'instruction_sep_context':
            # Instruction: {instruction}\nContext: {context}
            if context:
                # 截斷 context 如果超過 max_context_chars
                if len(context) > self.max_context_chars:
                    context = context[:self.max_context_chars]
                return f"Instruction: {instruction}\nContext: {context}"
            return instruction
        else:
            raise ValueError(f"Unknown text_format: {self.text_format}")
    
    def load_data(self, data_path: str) -> Tuple[List[str], np.ndarray, List[dict]]:
        """載入並預處理資料"""
        logger.info(f"Loading data from {data_path}")
        
        texts = []
        labels = []
        raw_data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = self.prepare_text(item)
                texts.append(text)
                labels.append(item['label'])
                raw_data.append(item)
        
        # 確保 labels 為 int64 類型
        labels_array = np.array(labels, dtype=np.int64)
        
        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Label distribution: {np.bincount(labels_array)}")
        
        return texts, labels_array, raw_data
    
    def train(self, texts: List[str], labels: np.ndarray, test_size: float = 0.2, plot: bool = False, debug: bool = False):
        """訓練路由器"""
        # 分割訓練/測試集
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # 生成 embeddings
        logger.info("Encoding training texts...")
        train_embeddings = self.encoder.encode(
            X_train, 
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        logger.info("Encoding test texts...")
        test_embeddings = self.encoder.encode(
            X_test,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 訓練分類器
        logger.info("Training classifier...")
        self.classifier.fit(train_embeddings, y_train)
        
        # 評估
        train_score = self.classifier.score(train_embeddings, y_train)
        test_score = self.classifier.score(test_embeddings, y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # 詳細評估
        y_pred = self.classifier.predict(test_embeddings)
        y_pred_proba = self.classifier.predict_proba(test_embeddings)
        
        # 計算 F1 score
        test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
        test_f1_macro = f1_score(y_test, y_pred, average='macro')
        test_f1_binary = f1_score(y_test, y_pred, pos_label=1, average='binary')
        
        # 計算 grey_zone_rate (0.45 < p_slow < 0.55)
        p_slow = y_pred_proba[:, 1]
        grey_zone = np.sum((p_slow > 0.45) & (p_slow < 0.55))
        grey_zone_rate = grey_zone / len(p_slow)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n" + "="*50)
        print("Classification Report")
        print("="*50)
        print(classification_report(
            y_test, y_pred, 
            target_names=['Fast Path (0)', 'Slow Path (1)']
        ))
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy: {test_score:.4f}")
        print(f"  F1 (weighted): {test_f1_weighted:.4f}")
        print(f"  F1 (macro): {test_f1_macro:.4f}")
        print(f"  F1 (binary, pos_label=1): {test_f1_binary:.4f}")
        print(f"  Grey Zone Rate: {grey_zone_rate:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  [[{cm[0,0]:5d}  {cm[0,1]:5d}]")
        print(f"   [{cm[1,0]:5d}  {cm[1,1]:5d}]]")
        
        # 可選的繪圖
        if plot:
            self._plot_confusion_matrix(cm)
        
        # 基本 metrics（總是回傳）
        metrics = {
            'train_acc': train_score,
            'test_acc': test_score,
            'test_f1_weighted': test_f1_weighted,
            'test_f1_macro': test_f1_macro,
            'test_f1_binary': test_f1_binary,
            'grey_zone_rate': grey_zone_rate,
            'confusion_matrix': cm.tolist(),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'label_distribution_train': np.bincount(y_train).tolist(),
            'label_distribution_test': np.bincount(y_test).tolist(),
            'label_distribution_all': np.bincount(labels).tolist(),
        }
        
        # Debug 模式才回傳大物件
        if debug:
            metrics.update({
                'test_embeddings': test_embeddings,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            })
        
        return metrics
    
    def predict_batch(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批次預測
        
        Returns:
            labels: 預測的標籤 (0 or 1, int)
            p_slow: Slow Path 的機率 (0.0 ~ 1.0)
            confidence_margin: 信心度邊際 (0.0 ~ 1.0, 越大越有把握)
        """
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        probas = self.classifier.predict_proba(embeddings)
        labels = np.argmax(probas, axis=1).astype(int)
        p_slow = probas[:, 1]  # Slow Path (label=1) 的機率
        confidence_margin = np.abs(p_slow - 0.5) * 2  # 0~1, 越大越有把握
        
        return labels, p_slow, confidence_margin
    
    def predict_single(self, text: str) -> Tuple[int, float, float]:
        """單一預測（內部會轉為 batch）"""
        labels, p_slow, confidence_margin = self.predict_batch([text])
        return int(labels[0]), float(p_slow[0]), float(confidence_margin[0])
    
    def save(self, save_dir: str = 'models'):
        """儲存模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 建立 artifact 包含所有必要資訊
        artifact = {
            "model_name": self.model_name,
            "classifier": self.classifier,
            "normalize": True,
            "text_format": self.text_format,
            "max_context_chars": self.max_context_chars,
        }
        
        # 儲存為單一 router.pkl 檔案
        router_path = save_path / 'router.pkl'
        with open(router_path, 'wb') as f:
            pickle.dump(artifact, f)
        
        logger.info(f"Model saved to {router_path}")
        
    def load(self, save_dir: str = 'models'):
        """載入模型"""
        save_path = Path(save_dir)
        
        # 載入 artifact
        router_path = save_path / 'router.pkl'
        with open(router_path, 'rb') as f:
            artifact = pickle.load(f)
        
        # 從 artifact 恢復所有資訊
        self.model_name = artifact["model_name"]
        self.classifier = artifact["classifier"]
        self.text_format = artifact.get("text_format", "instruction_sep_context")
        self.max_context_chars = artifact.get("max_context_chars", 2000)
        normalize = artifact.get("normalize", True)
        
        # 重新載入 encoder（因為 SentenceTransformer 不適合直接 pickle）
        logger.info(f"Loading SentenceTransformer: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)
        
        logger.info(f"Model loaded from {router_path}")
        logger.info(f"Text format: {self.text_format}, Max context chars: {self.max_context_chars}, Normalize: {normalize}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """繪製混淆矩陣（需要 matplotlib 和 seaborn）"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib or seaborn not available, skipping plot")
            return
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Fast (0)', 'Slow (1)'],
            yticklabels=['Fast (0)', 'Slow (1)']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        logger.info("Confusion matrix saved to confusion_matrix.png")
        plt.close()
    
    def save_metrics(self, metrics: dict, save_dir: str = 'models'):
        """儲存訓練 metrics 到 JSON 檔案"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        metrics_dict = {
            'model_name': self.model_name,
            'text_format': self.text_format,
            'max_context_chars': self.max_context_chars,
            'normalize': True,
            'train_size': metrics.get('train_size'),
            'test_size': metrics.get('test_size'),
            'label_distribution': {
                'all': metrics.get('label_distribution_all'),
                'train': metrics.get('label_distribution_train'),
                'test': metrics.get('label_distribution_test'),
            },
            'test_acc': metrics.get('test_acc'),
            'test_f1_macro': metrics.get('test_f1_macro'),
            'test_f1_weighted': metrics.get('test_f1_weighted'),
            'test_f1_binary': metrics.get('test_f1_binary'),
            'confusion_matrix': metrics.get('confusion_matrix'),
            'grey_zone_rate': metrics.get('grey_zone_rate'),
        }
        
        metrics_path = save_path / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics saved to {metrics_path}")


def run_ablation_experiments(data_path: str = 'dolly_processed.jsonl', heavy: bool = False, plot: bool = False, C: float = 1.0):
    """
    執行 ablation 實驗，比較不同配置的效果
    
    Ablation 項目:
    1. Text format: instruction_only vs instruction_sep_context
    2. Encoder: all-MiniLM-L6-v2 vs all-mpnet-base-v2 (only if heavy=True)
    
    Args:
        data_path: 資料路徑
        heavy: 如果 True，會跑四組實驗（含 mpnet）；如果 False，只跑 MiniLM 的兩種 text_format
        plot: 是否繪製混淆矩陣
        C: Logistic Regression 的正則化參數（default: 1.0）
    """
    results = []
    
    # 實驗配置
    if heavy:
        experiments = [
            {
                'model_name': 'all-MiniLM-L6-v2',
                'text_format': 'instruction_only',
                'name': 'MiniLM-L6-v2 + instruction_only'
            },
            {
                'model_name': 'all-MiniLM-L6-v2',
                'text_format': 'instruction_sep_context',
                'name': 'MiniLM-L6-v2 + instruction_sep_context'
            },
            {
                'model_name': 'all-mpnet-base-v2',
                'text_format': 'instruction_only',
                'name': 'mpnet-base-v2 + instruction_only'
            },
            {
                'model_name': 'all-mpnet-base-v2',
                'text_format': 'instruction_sep_context',
                'name': 'mpnet-base-v2 + instruction_sep_context'
            },
        ]
    else:
        # 預設只跑 MiniLM 的兩種 text_format
        experiments = [
            {
                'model_name': 'all-MiniLM-L6-v2',
                'text_format': 'instruction_only',
                'name': 'MiniLM-L6-v2 + instruction_only'
            },
            {
                'model_name': 'all-MiniLM-L6-v2',
                'text_format': 'instruction_sep_context',
                'name': 'MiniLM-L6-v2 + instruction_sep_context'
            },
        ]
    
    print("\n" + "="*80)
    print("ABLATION EXPERIMENTS")
    print("="*80)
    
    for exp_config in experiments:
        print(f"\n{'='*80}")
        print(f"Experiment: {exp_config['name']}")
        print(f"{'='*80}")
        
        # 創建 router
        router = SemanticRouter(
            model_name=exp_config['model_name'],
            text_format=exp_config['text_format'],
            max_context_chars=2000,
            C=C
        )
        
        # 載入資料
        texts, labels, _ = router.load_data(data_path)
        
        # 訓練
        metrics = router.train(texts, labels, test_size=0.2, plot=plot)
        
        # 儲存模型（使用實驗名稱作為目錄）
        exp_name_safe = exp_config['name'].replace(' ', '_').replace('+', '_').replace('-', '_')
        save_dir = f"models/{exp_name_safe}"
        router.save(save_dir)
        router.save_metrics(metrics, save_dir)
        
        # 儲存結果
        result = {
            'experiment': exp_config['name'],
            'model_name': exp_config['model_name'],
            'text_format': exp_config['text_format'],
            'test_acc': metrics['test_acc'],
            'test_f1_weighted': metrics['test_f1_weighted'],
            'test_f1_macro': metrics['test_f1_macro'],
            'test_f1_binary': metrics['test_f1_binary'],
            'grey_zone_rate': metrics['grey_zone_rate'],
        }
        results.append(result)
        
        print(f"\nResults for {exp_config['name']}:")
        print(f"  Test Accuracy: {metrics['test_acc']:.4f}")
        print(f"  Test F1 (weighted): {metrics['test_f1_weighted']:.4f}")
        print(f"  Test F1 (macro): {metrics['test_f1_macro']:.4f}")
        print(f"  Test F1 (binary): {metrics['test_f1_binary']:.4f}")
        print(f"  Grey Zone Rate: {metrics['grey_zone_rate']:.4f}")
    
    # 總結所有實驗結果
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<50} {'Test Acc':<12} {'F1 (weighted)':<15} {'F1 (macro)':<15}")
    print("-"*80)
    for result in results:
        print(f"{result['experiment']:<50} {result['test_acc']:<12.4f} {result['test_f1_weighted']:<15.4f} {result['test_f1_macro']:<15.4f}")
    
    # 儲存結果到 JSON
    results_path = Path('ablation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Ablation results saved to {results_path}")
    
    return results


def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Semantic Router')
    parser.add_argument('--data', type=str, default='dolly_processed.jsonl')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--ablation-heavy', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'])
    parser.add_argument('--text-format', type=str, default='instruction_sep_context', choices=['instruction_only', 'instruction_sep_context'])
    parser.add_argument('--max-context-chars', type=int, default=2000)
    parser.add_argument('--C', type=float, default=1.5)
    parser.add_argument('--save-dir', type=str, default='models')
    
    return parser.parse_args()


def main():
    """主函式：執行訓練或 ablation 實驗"""
    # 設置全局隨機種子以確保可重現性
    set_global_seed(42)
    
    args = parse_args()
    
    if args.ablation or args.ablation_heavy:
        # 執行 ablation 實驗
        run_ablation_experiments(args.data, heavy=args.ablation_heavy, plot=args.plot, C=args.C)
    else:
        # 單一實驗
        router = SemanticRouter(
            model_name=args.model,
            text_format=args.text_format,
            max_context_chars=args.max_context_chars,
            C=args.C
        )
        
        texts, labels, _ = router.load_data(args.data)
        metrics = router.train(texts, labels, test_size=0.2, plot=args.plot)
        
        router.save(args.save_dir)
        router.save_metrics(metrics, args.save_dir)
        
        print(f"\nFinal Results:")
        print(f"  Test Accuracy: {metrics['test_acc']:.4f}")
        print(f"  Test F1 (weighted): {metrics['test_f1_weighted']:.4f}")
        print(f"  Test F1 (macro): {metrics['test_f1_macro']:.4f}")
        print(f"  Test F1 (binary): {metrics['test_f1_binary']:.4f}")
        print(f"  Grey Zone Rate: {metrics['grey_zone_rate']:.4f}")


if __name__ == "__main__":
    main()