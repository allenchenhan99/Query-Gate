"""
test_router.py
Smoke test for SemanticRouter.

Usage:
  # 自動使用最佳模型（從 models 目錄的子目錄中尋找）
  python test_router.py --model-dir models
  
  # 指定特定模型目錄
  python test_router.py --model-dir models/MiniLM_L6_v2___instruction_sep_context
"""

import argparse
from pathlib import Path

from train_router import SemanticRouter


def parse_args():
    p = argparse.ArgumentParser(description="Smoke test for SemanticRouter")
    p.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory that contains router.pkl or parent directory with model subdirs (default: models). "
             "If multiple models found, will use the best one automatically.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    # 檢查是否直接指定了包含 router.pkl 的目錄
    router_path = model_dir / "router.pkl"
    
    # 如果不存在，嘗試在子目錄中尋找
    if not router_path.exists():
        # 尋找所有包含 router.pkl 的子目錄
        subdirs = [d for d in model_dir.iterdir() if d.is_dir() and (d / "router.pkl").exists()]
        
        if not subdirs:
            raise FileNotFoundError(
                f"router.pkl not found under: {model_dir.resolve()}\n"
                f"Available model directories:\n" + 
                "\n".join([f"  - {d.name}" for d in model_dir.iterdir() if d.is_dir()])
            )
        
        # 如果有多個模型，選擇最佳模型（MiniLM_L6_v2___instruction_sep_context）
        # 或讓用戶選擇
        if len(subdirs) > 1:
            # 優先選擇最佳模型
            best_model = "MiniLM_L6_v2___instruction_sep_context"
            best_path = model_dir / best_model
            if best_path.exists() and (best_path / "router.pkl").exists():
                model_dir = best_path
                print(f"Multiple models found. Using best model: {best_model}")
            else:
                # 如果最佳模型不存在，使用第一個找到的
                model_dir = subdirs[0]
                print(f"Multiple models found. Using: {model_dir.name}")
                print(f"Available models: {', '.join([d.name for d in subdirs])}")
        else:
            model_dir = subdirs[0]
            print(f"Using model from: {model_dir.name}")
    
    router = SemanticRouter()
    router.load(str(model_dir))

    tests = [
        # Fast (0) - classification-like (更像 Dolly)
        "Which of the following is useful for transportation: a lamp, a train, an apple, a bicycle?",
        "Identify which instrument is string or woodwind: Panduri, Zurna",
        "Which is a species of fish? Tope or Rope",

        # Fast (0) - summarization-like (要有長 context)
        "Instruction: Please summarize the following passage in 3 bullet points.\n"
        "Context: LinkedIn is a business and employment-focused social media platform used for professional networking and career development. "
        "It allows job seekers to post resumes and employers to post job opportunities. Users can create profiles, connect with others, "
        "join groups, publish articles, and share content. LinkedIn is owned by Microsoft and has hundreds of millions of members worldwide.",

        # Slow (1) - general QA
        "How do I start running?",
        "What is a REST API? Explain with an example.",

        # Slow (1) - creative writing
        "Write a short poem about the ocean in 8 lines.",
    ]

    print("\n=== SemanticRouter Smoke Test ===")
    print(f"Model dir: {model_dir.resolve()}\n")

    for text in tests:
        label, p_slow, margin = router.predict_single(text)
        route = "Fast" if label == 0 else "Slow"
        preview = (text[:72] + "...") if len(text) > 75 else text
        print(f"{preview:<75} -> {route:4s}  (p_slow={p_slow:.3f}, margin={margin:.3f})")

    print("\nSmoke test completed.\n")


if __name__ == "__main__":
    main()
