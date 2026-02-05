"""
Sentence Transformers + Logistic Regression
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
    """set global random seed to ensure reproducibility"""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # try to make GPU deterministic (might be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 2.x is available (will throw error if unsupported operator is encountered)
        # torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class SemanticRouter:
    """
    Semantic Router: text classifier based on embedding
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
            model_name: SentenceTransformer model name
                推薦選項:
                - 'all-MiniLM-L6-v2': lightweight fast (80MB, 384 dim)
                - 'all-mpnet-base-v2': better but slower (420MB, 768 dim)
                - 'paraphrase-multilingual-MiniLM-L12-v2': supports Chinese
            text_format: text format options
                - 'instruction_only': only use instruction
                - 'instruction_sep_context': instruction + [SEP] + context
            max_context_chars: maximum number of characters in context, truncated if exceeded
            C: Logistic Regression regularization parameter (C=1.0)
        """
        logger.info(f"Loading SentenceTransformer: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # handle class imbalance
        )
        self.model_name = model_name
        self.text_format = text_format
        self.max_context_chars = max_context_chars
        
    def prepare_text(self, item: dict) -> str:
        """
        convert data to text that can be used for embedding
        
        Args:
            item: dictionary containing instruction and context
            
        Returns:
            processed text string
        """
        instruction = item.get('instruction', '').strip()
        context = item.get('context', '').strip()
        
        # decide format based on text_format
        if self.text_format == 'instruction_only':
            # only use instruction
            return instruction
        elif self.text_format == 'instruction_sep_context':
            # Instruction: {instruction}\nContext: {context}
            if context:
                # truncate context if exceeds max_context_chars
                if len(context) > self.max_context_chars:
                    context = context[:self.max_context_chars]
                return f"Instruction: {instruction}\nContext: {context}"
            return instruction
        else:
            raise ValueError(f"Unknown text_format: {self.text_format}")
    
    def load_data(self, data_path: str) -> Tuple[List[str], np.ndarray, List[dict]]:
        """load and preprocess data"""
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
        
        # ensure labels are int64 type
        labels_array = np.array(labels, dtype=np.int64)
        
        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Label distribution: {np.bincount(labels_array)}")
        
        return texts, labels_array, raw_data
    
    def train(self, texts: List[str], labels: np.ndarray, test_size: float = 0.2, plot: bool = False, debug: bool = False):
        """train router"""
        # split training/test set
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # generate embeddings
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
        
        # train classifier
        logger.info("Training classifier...")
        self.classifier.fit(train_embeddings, y_train)
        
        # evaluate
        train_score = self.classifier.score(train_embeddings, y_train)
        test_score = self.classifier.score(test_embeddings, y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Test accuracy: {test_score:.4f}")
        
        # detailed evaluation
        y_pred = self.classifier.predict(test_embeddings)
        y_pred_proba = self.classifier.predict_proba(test_embeddings)
        
        # calculate F1 score
        test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
        test_f1_macro = f1_score(y_test, y_pred, average='macro')
        test_f1_binary = f1_score(y_test, y_pred, pos_label=1, average='binary')
        
        # calculate grey_zone_rate (0.45 < p_slow < 0.55)
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
        
        # optional plot
        if plot:
            self._plot_confusion_matrix(cm)
        
        # basic metrics (always return)
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
        
        # return large object in debug mode
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
        batch prediction
        
        Returns:
            labels: predicted labels (0 or 1, int)
            p_slow: probability of Slow Path (0.0 ~ 1.0)
            confidence_margin: confidence margin (0.0 ~ 1.0,越大越有把握)
            larger means more confident
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
        p_slow = probas[:, 1]  # probability of Slow Path (label=1)
        confidence_margin = np.abs(p_slow - 0.5) * 2  # 0~1, 越大越有把握
        
        return labels, p_slow, confidence_margin
    
    def predict_single(self, text: str) -> Tuple[int, float, float]:
        """single prediction (internal will be converted to batch)"""
        labels, p_slow, confidence_margin = self.predict_batch([text])
        return int(labels[0]), float(p_slow[0]), float(confidence_margin[0])
    
    def save(self, save_dir: str = 'models'):
        """save model"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # create artifact containing all necessary information
        artifact = {
            "model_name": self.model_name,
            "classifier": self.classifier,
            "normalize": True,
            "text_format": self.text_format,
            "max_context_chars": self.max_context_chars,
        }
        
        # save as single router.pkl file
        router_path = save_path / 'router.pkl'
        with open(router_path, 'wb') as f:
            pickle.dump(artifact, f)
        
        logger.info(f"Model saved to {router_path}")
        
    def load(self, save_dir: str = 'models'):
        """load model"""
        save_path = Path(save_dir)
        
        # load artifact
        router_path = save_path / 'router.pkl'
        with open(router_path, 'rb') as f:
            artifact = pickle.load(f)
        
        # restore all information from artifact
        self.model_name = artifact["model_name"]
        self.classifier = artifact["classifier"]
        self.text_format = artifact.get("text_format", "instruction_sep_context")
        self.max_context_chars = artifact.get("max_context_chars", 2000)
        normalize = artifact.get("normalize", True)
        
        # reload encoder (because SentenceTransformer is not suitable for direct pickle)
        logger.info(f"Loading SentenceTransformer: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)
        
        logger.info(f"Model loaded from {router_path}")
        logger.info(f"Text format: {self.text_format}, Max context chars: {self.max_context_chars}, Normalize: {normalize}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """plot confusion matrix (requires matplotlib and seaborn)"""
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
        """save training metrics to JSON file"""
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
    execute ablation experiments, compare different configurations
    
    Ablation items:
    1. Text format: instruction_only vs instruction_sep_context
    2. Encoder: all-MiniLM-L6-v2 vs all-mpnet-base-v2 (only if heavy=True)
    
    Args:
        data_path: data path
        heavy: if True, run four experiments (including mpnet); if False, only run two text_formats of MiniLM
        plot: whether to plot confusion matrix
        C: Logistic Regression regularization parameter (default: 1.0)
    """
    results = []
    
    # experiment configurations
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
        # default only run two text_formats of MiniLM
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
        
        # create router
        router = SemanticRouter(
            model_name=exp_config['model_name'],
            text_format=exp_config['text_format'],
            max_context_chars=2000,
            C=C
        )
        
        # load data
        texts, labels, _ = router.load_data(data_path)
        
        # train
        metrics = router.train(texts, labels, test_size=0.2, plot=plot)
        
        # save model (use experiment name as directory)
        exp_name_safe = exp_config['name'].replace(' ', '_').replace('+', '_').replace('-', '_')
        save_dir = f"models/{exp_name_safe}"
        router.save(save_dir)
        router.save_metrics(metrics, save_dir)
        
        # save results
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
    
    # summarize all experiment results
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<50} {'Test Acc':<12} {'F1 (weighted)':<15} {'F1 (macro)':<15}")
    print("-"*80)
    for result in results:
        print(f"{result['experiment']:<50} {result['test_acc']:<12.4f} {result['test_f1_weighted']:<15.4f} {result['test_f1_macro']:<15.4f}")
    
    # save results to JSON
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
    """main function: execute training or ablation experiments"""
    # set global random seed to ensure reproducibility
    set_global_seed(42)
    
    args = parse_args()
    
    if args.ablation or args.ablation_heavy:
        # execute ablation experiments
        run_ablation_experiments(args.data, heavy=args.ablation_heavy, plot=args.plot, C=args.C)
    else:
        # single experiment
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