"""
FastAPI Application: API Gateway + Runtime Orchestrator (Version 2)
Improved Decision Policy: Load-Aware Dynamic Thresholding

Improvements:
1. Use Semaphore to track the actual load of Slow Path (not Queue Size)
2. Dynamic threshold adjustment (Quadratic Penalty)
3. Cache model outputs, re-decide each time
"""

import asyncio
import time
import random
import argparse
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np

from app.service import load_router, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# global variable: model directory (read from command line argument or environment variable)
MODEL_DIR: Optional[str] = None

# ==================== Configuration ====================
BATCH_WINDOW_MS = 50  # batching window (milliseconds)
MAX_BATCH_SIZE = 32  # 最大批次大小
CACHE_SIZE = 10000  # LRU Cache size

# [New] Decision Policy Parameters
BASE_THRESHOLD = 0.5        # base threshold (Cost-Weighting)
ADAPTIVE_ALPHA = 0.45       # load sensitivity (Sensitivity)
MAX_SLOW_CONCURRENCY = 50   # maximum concurrency of Slow Path (Load-Awareness)
HIGH_CONFIDENCE_MARGIN = 0.3  # high confidence threshold

# Global State
app = FastAPI(title="Semantic Router API v2", version="2.0.0")

# Batching queue
batch_queue: asyncio.Queue = None
batch_worker_task: asyncio.Task = None

# LRU Cache: text -> (model_label, p_slow, margin)
# note: Cache stores model outputs, not decision results
cache: OrderedDict = OrderedDict()

# [New] Semaphore for Slow Path load tracking (thread-safe)
SLOW_PATH_SEMAPHORE: asyncio.Semaphore = None


# Request/Response Models
class QueryRequest(BaseModel):
    text: str


# Argument Parsing
def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description='Semantic Router API Server v2')
    parser.add_argument('--model-dir', type=str, default='models/MiniLM_L6_v2___instruction_sep_context')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    return parser.parse_args()


# Helper Functions
def get_current_slow_load() -> int:
    """get the current number of active Slow Path"""
    if SLOW_PATH_SEMAPHORE is None:
        return 0
    # Semaphore._value represents the remaining available number
    # active number = total number - remaining number
    return MAX_SLOW_CONCURRENCY - SLOW_PATH_SEMAPHORE._value


# Cache Management
def get_cache_key(text: str) -> str:
    """generate cache key (only strip, not lower, keep case distinction)"""
    return text.strip()


def get_from_cache(text: str) -> Optional[Tuple[int, float, float]]:
    """get model output from cache (not decision result)"""
    key = get_cache_key(text)
    if key in cache:
        # move to the end (LRU)
        cache.move_to_end(key)
        return cache[key]
    return None


def set_cache(text: str, model_label: int, p_slow: float, margin: float):
    """
    write to cache (store model output, not decision result)
    這樣可以在 Cache Hit 時重新應用決策邏輯
    """
    key = get_cache_key(text)
    cache[key] = (model_label, p_slow, margin)

    # limit cache size
    if len(cache) > CACHE_SIZE:
        cache.popitem(last=False)  # remove the oldest item


# Decision Policy
def apply_decision_policy(
    model_label: int,
    p_slow: float,
    margin: float
) -> Tuple[int, str, float]:
    """
    Decision Policy: Load-Aware Dynamic Thresholding

    Args:
        model_label: model predicted label (0 or 1)
        p_slow: probability of Slow Path (0.0 ~ 1.0)
        margin: confidence margin (0.0 ~ 1.0)

    Returns:
        final_label: final decision (0 or 1)
        decision_type: decision reason ('high_confidence', 'slow_granted', 'load_shedding', 'fast_default')
        threshold_used: used dynamic threshold
    """
    # 1. high confidence -> use model prediction directly (avoid over-intervention)
    if margin > HIGH_CONFIDENCE_MARGIN:
        return model_label, "high_confidence", 0.0

    # 2. low confidence + Slow Path prediction -> check load
    if model_label == 1 or p_slow > 0.5:
        # calculate current load ratio
        current_active = get_current_slow_load()
        load_ratio = min(1.0, current_active / MAX_SLOW_CONCURRENCY)

        # dynamic threshold (quadratic penalty)
        # formula: T = Base + Alpha * (Load^2)
        # the higher the load, the higher the threshold (harder to enter Slow Path)
        dynamic_threshold = BASE_THRESHOLD + ADAPTIVE_ALPHA * (load_ratio ** 2)

        # decide whether to allow entry into Slow Path
        if p_slow > dynamic_threshold:
            return 1, "slow_granted", dynamic_threshold
        else:
            # downgrade to Fast Path
            return 0, "load_shedding", dynamic_threshold

    # 3. Fast Path prediction
    return 0, "fast_default", 0.0


# Batching System
class BatchItem:
    """batch item: contains text, future and timestamp"""
    def __init__(self, text: str, future: asyncio.Future):
        self.text = text
        self.future = future
        self.timestamp = time.time()


async def batch_worker():
    """background batch processing worker"""
    logger.info("Batch worker started")

    while True:
        try:
            batch_items: List[BatchItem] = []
            batch_texts: List[str] = []

            # wait for the first item (ensure there is work to start)
            first_item = await batch_queue.get()
            batch_items.append(first_item)
            batch_texts.append(first_item.text)

            # set deadline
            deadline = time.time() + (BATCH_WINDOW_MS / 1000.0)

            # use get_nowait to quickly fill the batch within the time window
            while len(batch_items) < MAX_BATCH_SIZE:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break

                try:
                    # non-blocking get (if queue has something, get it immediately)
                    item = batch_queue.get_nowait()
                    batch_items.append(item)
                    batch_texts.append(item.text)
                except asyncio.QueueEmpty:
                    # queue is empty, wait for a short time and check again
                    await asyncio.sleep(0.001)
                    # if already past deadline, stop collecting
                    if time.time() >= deadline:
                        break

            # if there is a batch, perform inference
            if batch_items:
                logger.debug(f"Processing batch of {len(batch_items)} items")

                try:
                    # call service.predict for batch inference
                    labels, p_slows, margins, latency_ms = predict(batch_texts)

                    logger.info(f"Processed batch size={len(batch_items)} latency_ms={latency_ms:.2f}")

                    # [Modified] write the results back to each future (store model output, not decision result)
                    for i, item in enumerate(batch_items):
                        model_label = int(labels[i])
                        p_slow = float(p_slows[i])
                        margin = float(margins[i])

                        # set future result (check future status to avoid InvalidStateError)
                        if not item.future.done():
                            item.future.set_result({
                                'model_label': model_label,  # model original prediction
                                'p_slow': p_slow,
                                'margin': margin,
                                'latency_ms': latency_ms / len(batch_items)  # average latency
                            })

                except Exception as e:
                    logger.error(f"Batch processing error: {e}", exc_info=True)
                    # set all futures to exception when error occurs (check status)
                    for item in batch_items:
                        if not item.future.done():
                            item.future.set_exception(e)

        except Exception as e:
            logger.error(f"Batch worker error: {e}", exc_info=True)
            await asyncio.sleep(0.1)


# FastAPI Endpoints
@app.post("/v1/query-process")
async def query_process(request: QueryRequest):
    """
    process a single query request (Version 2 with Dynamic Thresholding)

    Process:
    1. check cache (store model output)
    2. Cache hit → re-decide; Cache miss → batch inference
    3. apply decision policy
    4. simulate Fast/Slow Path execution (use Semaphore to track load)
    5. return result + latency header

    Response: {"label": "0"} or {"label": "1"} (符合題目規格)
    """
    start_time = time.perf_counter()

    # variable to store the result
    final_label = None
    router_latency_ms = 0.0
    decision_reason = ""
    threshold_used = 0.0

    # 1. check cache
    cached_result = get_from_cache(request.text)
    if cached_result is not None:
        # CACHE HIT branch: get model output, re-decide
        model_label, p_slow, margin = cached_result
        final_label, decision_reason, threshold_used = apply_decision_policy(
            model_label, p_slow, margin
        )
        final_label = str(final_label)

        # calculate Router Latency for Cache Hit
        router_end_time = time.perf_counter()
        router_latency_ms = (router_end_time - start_time) * 1000.0

        logger.info(
            f"CACHE_HIT start_time={start_time:.6f} router_end_time={router_end_time:.6f} "
            f"router_latency_ms={router_latency_ms:.6f} decision={decision_reason}"
        )
    else:
        # CACHE MISS branch
        logger.info(f"CACHE_MISS start_time={start_time:.6f}")

        # 2. Cache miss → enqueue to batching queue
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await batch_queue.put(BatchItem(request.text, future))

        # 3. wait for the batch result
        try:
            result = await future
            model_label = result['model_label']
            p_slow = result['p_slow']
            margin = result['margin']

            # apply decision policy
            final_label, decision_reason, threshold_used = apply_decision_policy(
                model_label, p_slow, margin
            )
            final_label = str(final_label)

            # Cache store model output (not decision result)
            set_cache(request.text, model_label, p_slow, margin)

            # use total latency (including queueing + inference)
            router_end_time = time.perf_counter()
            router_latency_ms = (router_end_time - start_time) * 1000.0

            logger.info(
                f"CACHE_MISS start_time={start_time:.6f} router_end_time={router_end_time:.6f} "
                f"router_latency_ms={router_latency_ms:.6f} decision={decision_reason}"
            )
        except asyncio.CancelledError:
            logger.warning("Request cancelled")
            raise HTTPException(status_code=499, detail="Request cancelled")
        except Exception as e:
            logger.error(f"Query processing error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    # 4. simulate the execution time of Fast/Slow Path
    # [Modified] use Semaphore to track Slow Path load
    if final_label == "0":  # Fast Path
        sleep_time = random.uniform(0.01, 0.2)
        await asyncio.sleep(sleep_time)
        logger.debug(f"Fast Path simulation: slept {sleep_time:.3f}s")
    else:  # Slow Path
        # use Semaphore to manage concurrency (thread-safe)
        async with SLOW_PATH_SEMAPHORE:
            sleep_time = random.uniform(1.0, 3.0)
            await asyncio.sleep(sleep_time)
            logger.debug(f"Slow Path simulation: slept {sleep_time:.3f}s")

    # 5. return the result
    # note: header only returns "Router decision time", not including the sleep time
    latency_str = "{:.2f}".format(router_latency_ms)

    response = JSONResponse(content={"label": final_label})
    response.headers["x-router-latency"] = latency_str
    response.headers["x-system-load"] = str(get_current_slow_load())  # Debug: current load
    response.headers["x-decision-type"] = decision_reason  # Debug: decision reason
    if threshold_used > 0:
        response.headers["x-threshold"] = f"{threshold_used:.2f}"  # Debug: used threshold
    return response


@app.get("/health")
async def health():
    """health check"""
    return {
        "status": "healthy",
        "queue_size": batch_queue.qsize() if batch_queue else 0,
        "cache_size": len(cache),
        "slow_path_active": get_current_slow_load(),
        "slow_path_capacity": MAX_SLOW_CONCURRENCY
    }


# Startup/Shutdown
@app.on_event("startup")
async def startup():
    """startup initialization"""
    global batch_queue, batch_worker_task, MODEL_DIR, SLOW_PATH_SEMAPHORE

    logger.info("Starting up...")

    # load router (use command line argument first, then environment variable, then default value)
    model_dir = MODEL_DIR if MODEL_DIR is not None else None
    load_router(model_dir)

    # initialize batching queue
    batch_queue = asyncio.Queue()

    # [New] initialize Semaphore (limit Slow Path concurrency)
    SLOW_PATH_SEMAPHORE = asyncio.Semaphore(MAX_SLOW_CONCURRENCY)

    # start batch worker
    batch_worker_task = asyncio.create_task(batch_worker())

    logger.info(f"Startup complete (Decision Policy: Dynamic Thresholding, MAX_SLOW={MAX_SLOW_CONCURRENCY})")


@app.on_event("shutdown")
async def shutdown():
    """shutdown cleanup"""
    global batch_worker_task

    logger.info("Shutting down...")

    if batch_worker_task:
        batch_worker_task.cancel()
        try:
            await batch_worker_task
        except asyncio.CancelledError:
            pass

    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn

    # parse command line arguments
    args = parse_args()

    # set global MODEL_DIR (used at startup)
    MODEL_DIR = args.model_dir

    # start the service
    uvicorn.run(app, host=args.host, port=args.port)
