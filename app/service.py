"""
Router Engine Service Layer
Only handle "load model + batch inference", not handling HTTP、queue
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# add project root to path, to import train_router
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train_router import SemanticRouter

logger = logging.getLogger(__name__)

# global variable: Router instance (loaded once at startup)
ROUTER: SemanticRouter = None


def load_router(model_dir: Optional[str] = None) -> None:
    """
    load router at startup (only load once)
    
    Args:
        model_dir: model directory path. if None, read from environment variable MODEL_DIR, default is "models"
    """
    global ROUTER
    
    if ROUTER is not None:
        logger.warning("Router already loaded, skipping reload")
        return
    
    # get model path from environment variable or parameter
    if model_dir is None:
        model_dir = os.getenv("MODEL_DIR", "models")
    
    logger.info(f"Loading router from: {model_dir}")
    model_path = Path(model_dir)
    
    # check if the root directory has router.pkl (Default part is Optional)
    router_pkl = model_path / "router.pkl"
    
    if not router_pkl.exists():
        # if the root directory doesn't have router.pkl, automatically find the best model in the subdirectories
        logger.info(f"router.pkl not found in {model_path}, searching subdirectories...")
        subdirs = [d for d in model_path.iterdir() if d.is_dir() and (d / "router.pkl").exists()]
        
        if not subdirs:
            # list all available model directories
            all_dirs = [d.name for d in model_path.iterdir() if d.is_dir()] if model_path.exists() else []
            raise FileNotFoundError(
                f"router.pkl not found at: {router_pkl.resolve()}\n"
                f"Available subdirectories: {all_dirs}\n"
                f"Please set MODEL_DIR environment variable to a specific model directory, e.g.:\n"
                f"  export MODEL_DIR=models/MiniLM_L6_v2___instruction_sep_context"
            )
        
        # first choose the best model (if exists)
        best_model_name = "MiniLM_L6_v2___instruction_sep_context"
        best_path = model_path / best_model_name
        if best_path.exists() and (best_path / "router.pkl").exists():
            model_path = best_path
            logger.info(f"Using best model: {best_model_name}")
        else:
            # use the first found model
            model_path = subdirs[0]
            logger.info(f"Using model from: {model_path.name}")
            logger.info(f"Available models: {[d.name for d in subdirs]}")
    
    ROUTER = SemanticRouter()
    ROUTER.load(str(model_path))
    logger.info(f"Router loaded successfully from: {model_path.resolve()}")


def predict(texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    batch inference API
    
    Args:
        texts: input text list
        
    Returns:
        labels: predicted labels (0 or 1, int array)
        p_slow: probability of Slow Path (label=1) (0.0 ~ 1.0, float array)
        margin: confidence margin (0.0 ~ 1.0, float array) -> the larger the margin, the more confident the prediction
        latency_ms: inference latency (milliseconds)
    """
    if ROUTER is None:
        raise RuntimeError("Router not loaded. Call load_router() first.")
    
    if not texts:
        return np.array([]), np.array([]), np.array([]), 0.0
    
    # start the timer
    start_time = time.perf_counter()
    
    # batch inference
    labels, p_slow, margin = ROUTER.predict_batch(texts)
    
    # calculate the latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return labels, p_slow, margin, latency_ms


def get_router() -> SemanticRouter:
    """get the loaded router instance (used for testing or advanced operations)"""
    if ROUTER is None:
        raise RuntimeError("Router not loaded. Call load_router() first.")
    return ROUTER
