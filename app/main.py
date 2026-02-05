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

MODEL_DIR: Optional[str] = None

# Configuration
BATCH_WINDOW_MS = 50  # batching window (ms)
MAX_BATCH_SIZE = 32  # max batch size
CACHE_SIZE = 10000  # LRU Cache size
GREY_ZONE_LOW = 0.45  # grey zone low
GREY_ZONE_HIGH = 0.55  # grey zone high

# Global State
app = FastAPI(title="Semantic Router API", version="1.0.0")

# Batching queue
batch_queue: asyncio.Queue = None
batch_worker_task: asyncio.Task = None

# LRU Cache: text -> (label, p_slow, margin)
cache: OrderedDict = OrderedDict()


# Request/Response Models
class QueryRequest(BaseModel):
    text: str
# Response only return {"label": "0"} or {"label": "1"}

# Argument Parsing
def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Semantic Router API Server')
    parser.add_argument('--model-dir', type=str, default='models/MiniLM_L6_v2___instruction_sep_context')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    return parser.parse_args()


# Cache Management
def get_cache_key(text: str) -> str:
    """生成 cache key（只 strip，不 lower，保持大小寫區分）"""
    return text.strip()


def get_from_cache(text: str) -> Optional[Tuple[int, float, float]]:
    """從 cache 取得結果"""
    key = get_cache_key(text)
    if key in cache:
        # 移到最後（LRU）
        cache.move_to_end(key)
        return cache[key]
    return None


def set_cache(text: str, label: int, p_slow: float, margin: float):
    """寫入 cache"""
    key = get_cache_key(text)
    cache[key] = (label, p_slow, margin)
    
    # 限制 cache 大小
    if len(cache) > CACHE_SIZE:
        cache.popitem(last=False)  # 移除最舊的


# Batching System
class BatchItem:
    """批次項目：包含 text、future 和 timestamp"""
    def __init__(self, text: str, future: asyncio.Future):
        self.text = text
        self.future = future
        self.timestamp = time.time()


async def batch_worker():
    """背景批次處理 worker（優化版本：先等第一個，再用 get_nowait 拉滿）"""
    logger.info("Batch worker started")
    
    while True:
        try:
            batch_items: List[BatchItem] = []
            batch_texts: List[str] = []
            
            # wait for the first item (ensure there is work to do)
            first_item = await batch_queue.get()
            batch_items.append(first_item)
            batch_texts.append(first_item.text)
            
            # set deadline
            deadline = time.time() + (BATCH_WINDOW_MS / 1000.0)
            
            # fill the batch within the time window using get_nowait
            while len(batch_items) < MAX_BATCH_SIZE:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                
                try:
                    # non-blocking get (if queue has items, get immediately)
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
                    
                    # write the results back to each future
                    for i, item in enumerate(batch_items):
                        label = int(labels[i])
                        p_slow = float(p_slows[i])
                        margin = float(margins[i])
                        
                        # apply decision policy
                        final_label = apply_decision_policy(
                            label, p_slow, margin, batch_queue.qsize()
                        )
                        
                        # write into cache
                        set_cache(item.text, final_label, p_slow, margin)
                        
                        # set future result (check future status to avoid InvalidStateError)
                        if not item.future.done():
                            item.future.set_result({
                                'label': final_label,
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


# Decision Policy
def apply_decision_policy(
    model_label: int, 
    p_slow: float, 
    margin: float, 
    queue_size: int
) -> int:
    """
    decision policy: based on model output + system load for final decision
    
    Args:
        model_label: model predicted label (0 or 1)
        p_slow: probability of Slow Path
        margin: confidence margin
        queue_size: current queue size
        
    Returns:
        final decision label (0 or 1)
    """
    # high confidence: use model prediction directly
    if margin > 0.3:  # high confidence threshold
        return model_label
    
    # grey zone decision: based on system load
    if GREY_ZONE_LOW < p_slow < GREY_ZONE_HIGH:
        # queue is large (high load) -> Fast Path (boost throughput) -> return 0
        if queue_size > 10:
            return 0
        # queue is small (low load) -> Slow Path (boost quality) -> return 1
        elif queue_size < 3:
            return 1
        # medium load -> use model prediction -> return model_label
        else:
            return model_label
    
    # non-grey zone: use model prediction
    return model_label


# FastAPI Endpoints
@app.post("/v1/query-process")
async def query_process(request: QueryRequest):
    """
    process a single query request
    
    process:
    1. check cache
    2. Cache miss -> enqueue to batching queue
    3. wait for the result
    4. return the result + latency header
    
    Response: {"label": "0"} or {"label": "1"}
    """
    start_time = time.perf_counter()
    
    # variable to store the result
    final_label = None
    router_latency_ms = 0.0
    
    # 1. check cache
    cached_result = get_from_cache(request.text)
    if cached_result is not None:
        # CACHE HIT branch
        label, _, _ = cached_result
        final_label = str(label)
        # calculate the Router Latency for Cache Hit
        router_end_time = time.perf_counter()
        router_latency_ms = (router_end_time - start_time) * 1000.0
        
        logger.info(
            f"CACHE_HIT start_time={start_time:.6f} router_end_time={router_end_time:.6f} "
            f"router_latency_ms={router_latency_ms:.6f} formatted={router_latency_ms:.2f}"
        )
    else:
        # CACHE MISS branch
        logger.info(f"CACHE_MISS start_time={start_time:.6f}")
        
        # 2. Cache miss -> enqueue to batching queue
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await batch_queue.put(BatchItem(request.text, future))
        
        # 3. wait for the batch result
        try:
            result = await future
            final_label = str(result['label'])
            # use the total latency (including queueing + inference)
            router_end_time = time.perf_counter()
            router_latency_ms = (router_end_time - start_time) * 1000.0
            
            logger.info(
                f"CACHE_MISS start_time={start_time:.6f} router_end_time={router_end_time:.6f} "
                f"router_latency_ms={router_latency_ms:.6f} formatted={router_latency_ms:.2f}"
            )
        except asyncio.CancelledError:
            logger.warning("Request cancelled")
            raise HTTPException(status_code=499, detail="Request cancelled")
        except Exception as e:
            logger.error(f"Query processing error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    # 4. simulate the execution time of Fast/Slow Path
    # this step should be done after the Router Latency is calculated
    if final_label == "0":  # Fast Path
        sleep_time = random.uniform(0.01, 0.2)
        await asyncio.sleep(sleep_time)
        logger.debug(f"Fast Path simulation: slept {sleep_time:.3f}s")
    else:  # Slow Path
        sleep_time = random.uniform(1.0, 3.0)
        await asyncio.sleep(sleep_time)
        logger.debug(f"Slow Path simulation: slept {sleep_time:.3f}s")
    
    # 5. return the result
    # note: header only returns "Router Latency", 
    #       not including the sleep time
    latency_str = "{:.2f}".format(router_latency_ms)
    
    response = JSONResponse(content={"label": final_label})
    response.headers["x-router-latency"] = latency_str
    return response


@app.get("/health")
async def health():
    """health check"""
    return {
        "status": "healthy",
        "queue_size": batch_queue.qsize() if batch_queue else 0,
        "cache_size": len(cache)
    }


# Startup/Shutdown
@app.on_event("startup")
async def startup():
    """startup initialization"""
    global batch_queue, batch_worker_task, MODEL_DIR
    
    logger.info("Starting up...")
    
    # load router (use command line argument first, then environment variable, then default value)
    model_dir = MODEL_DIR if MODEL_DIR is not None else None
    load_router(model_dir)
    
    # initialize batching queue
    batch_queue = asyncio.Queue()
    
    # start batch worker
    batch_worker_task = asyncio.create_task(batch_worker())
    
    logger.info("Startup complete")


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
    
    # set global MODEL_DIR (used in startup)
    MODEL_DIR = args.model_dir
    
    # start the service
    uvicorn.run(app, host=args.host, port=args.port)
