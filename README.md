# Query Gateway Microservices

A high-performance semantic router microservice that intelligently routes AI queries to Fast Path or Slow Path based on semantic complexity, with dynamic batching, caching, and load-aware decision making.

## Table of Contents

- [Quick Start with Docker](#quick-start-with-docker)
- [System Overview](#system-overview)
- [Architecture Design](#architecture-design)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [API Specification](#api-specification)
- [Development Setup](#development-setup)
- [System Design Details](#system-design-details)
- [Classification Performance](#classification-performance)

---

## Quick Start with Docker

**Note**: Models are not included in the Git repository. You need to train them first before building the Docker image.

### Prerequisites

- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- 4GB+ RAM available
- **Trained models in `models/` directory** (see [Training](#training-the-router-required))

### Build & Run

```bash
# 1. Build the Docker image
docker build -t query-gateway:latest .

# 2. Run the container
docker run -d \
  --name semantic-router \
  -p 8000:8000 \
  query-gateway:latest

# 3. Check if it's running
curl http://localhost:8000/health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "queue_size": 0,
  "cache_size": 0
}
```

### Test the API

```bash
# Test Fast Path query
curl -X POST http://localhost:8000/v1/query-process \
  -H "Content-Type: application/json" \
  -d '{"text":"Which is a species of fish? Tope or Rope"}'

# Expected: {"label": "0"}

# Test Slow Path query
curl -X POST http://localhost:8000/v1/query-process \
  -H "Content-Type: application/json" \
  -d '{"text":"Write a short poem about the ocean in 8 lines."}'

# Expected: {"label": "1"}
```

### Advanced Docker Usage

**Run with specific model**:
```bash
docker run -d \
  -p 8000:8000 \
  -e MODEL_DIR=models/MiniLM_L6_v2___instruction_sep_context \
  query-gateway:latest
```

**Run with logs**:
```bash
docker run -d \
  --name semantic-router \
  -p 8000:8000 \
  query-gateway:latest

# View logs
docker logs -f semantic-router
```

**Run Version 2 (Dynamic Thresholding)**:
```bash
docker run -d \
  -p 8000:8000 \
  query-gateway:latest \
  uvicorn app.main2:app --host 0.0.0.0 --port 8000
```

**Mount local models** (avoid rebuilding on model updates):
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  query-gateway:latest
```

### Stop & Clean Up

```bash
# Stop container
docker stop semantic-router

# Remove container
docker rm semantic-router

# Remove image
docker rmi query-gateway:latest
```

### Docker Image Info

- **Base Image**: `python:3.11-slim`
- **Image Size**: ~2.5GB (includes models)
- **Layers**: Multi-stage build for optimization
- **Health Check**: Built-in `/health` endpoint

---

## System Overview

This project implements a **Compound AI System** gateway that routes user queries to appropriate processing paths based on semantic complexity:

- **Fast Path (Label 0)**: Simple tasks like `classification` and `summarization`
- **Slow Path (Label 1)**: Complex tasks like `creative_writing` and `general_qa`

The system is designed to balance **latency**, **throughput**, and **cost** while handling high-concurrency requests.

### Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| HTTP Server | FastAPI + uvicorn | Async I/O, high performance |
| Semantic Router | Sentence-BERT + Logistic Regression | Semantic understanding & classification |
| Concurrency | asyncio | Non-blocking I/O |
| Batching | Time-window Queue | Request aggregation |
| Caching | LRU Cache (OrderedDict) | Reduce redundant computation |

---

## Architecture Design

### System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     HTTP Request                            │
│                  POST /v1/query-process                     │
│                   {"text": "..."}                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  Check Cache   │
            └────┬───────┬───┘
                 │       │
          [Hit]  │       │  [Miss]
                 │       │
                 ▼       ▼
         ┌───────────┐  ┌──────────────────┐
         │  Return   │  │ Enqueue to Batch │
         │  Result   │  │      Queue       │
         └───────────┘  └────────┬─────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Batch Worker         │
                    │ (Background Task)      │
                    │ - Collect requests     │
                    │   in 50ms window       │
                    │ - Max batch size: 32   │
                    └────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────────────┐
                    │   Batch Inference      │
                    │ encoder.encode(texts)  │
                    │ classifier.predict()   │
                    └────────┬───────────────┘
                             │
                             ▼
                    ┌────────────────────────┐
                    │  Decision Policy       │
                    │ - Cost-Weighting       │
                    │ - Confidence           │
                    │ - Load-Awareness       │
                    └────────┬───────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Response + Header   │
                 │ {"label": "0"}        │
                 │ x-router-latency: 45  │
                 └───────────────────────┘
```

### Key Components

#### 1. **FastAPI Application** (`app/main.py`)
- Handles HTTP requests asynchronously
- Manages batching queue and cache
- Implements decision policy

#### 2. **Service Layer** (`app/service.py`)
- Loads and manages the semantic router model
- Performs batch inference

#### 3. **Semantic Router** (`train_router.py`)
- Sentence Transformer encoder (all-MiniLM-L6-v2)
- Logistic Regression classifier
- Supports multiple text formats and ablation experiments

---

## Key Features

### 1. Semantic Router

**Implementation**: Embedding-based classifier

- **Encoder**: `all-MiniLM-L6-v2` (80MB, 384 dimensions)
- **Classifier**: Logistic Regression with balanced class weights
- **Text Format**: `instruction_sep_context` (includes instruction + context)

**Why this approach?**
- Strong semantic generalization (no keyword hard rules)
- Handles paraphrased or differently worded inputs
- Efficient batch encoding (10-20x faster than sequential)
- Probability outputs have interpretable confidence scores

### 2. High Concurrency & Async I/O

**Design**: FastAPI + asyncio

- All I/O operations are non-blocking (`async/await`)
- Main thread never blocks on inference
- `asyncio.Queue` for request aggregation
- Each request awaits its own `asyncio.Future` result

### 3. Dynamic Batching

**Configuration**:
```python
BATCH_WINDOW_MS = 50    # Batch collection time window (milliseconds)
MAX_BATCH_SIZE = 32     # Maximum batch size
```

**Mechanism**:
1. Wait for the first request (blocking wait)
2. Set deadline = now + 50ms
3. Quickly collect more requests using `get_nowait()` (non-blocking)
4. Trigger inference when:
   - Collected 32 requests (MAX_BATCH_SIZE), OR
   - Time window expires (50ms)
5. Perform batch inference once: `encoder.encode(batch_texts)`
6. Distribute results back to individual futures

**Why 50ms window?**
- Trade-off between latency and throughput
- Ensures P95 latency < 100ms while maximizing batch efficiency
- In high traffic: batches fill up quickly (batch size → 32)
- In low traffic: still responsive (max wait = 50ms)

**Performance Impact**:
- Batch inference (32 requests): ~30ms
- Sequential inference (32 requests): ~300ms
- **10x throughput improvement**

### 4. Caching Mechanism

**Implementation**: LRU Cache (Least Recently Used)

```python
CACHE_SIZE = 10000  # Maximum cached entries
cache: OrderedDict = OrderedDict()
```

**Cache Strategy**:
- **Key**: `text.strip()` (preserves case sensitivity)
- **Value**: `(label, p_slow, confidence_margin)`
- Automatically evicts least recently used entries when full

**Performance Impact**:
- Cache Hit: ~0.01ms (measured)
- Cache Miss: ~110ms (measured)
- **Speed improvement: ~10,000x faster** for cached queries
- Expected cache hit rate: 20-30% (for repeated queries like FAQs)
- With 30% cache hit rate: Expected P50 latency ~70-80ms (vs 110ms without cache)

### 5. Decision Policy (Bonus)

**Problem**: Cannot rely solely on model prediction. Must balance:
1. **Cost-Weighting**: Fast Path is cheap, Slow Path is expensive
2. **Confidence**: Router's confidence score (grey zone uncertainty)
3. **Load-Awareness**: Current system load (queue size)

**Algorithm**:

```python
def apply_decision_policy(model_label, p_slow, margin, queue_size):
    # 1. High Confidence → Use model prediction directly
    if margin > 0.3:  # margin = |p_slow - 0.5| * 2
        return model_label

    # 2. Grey Zone (0.45 < p_slow < 0.55) → Consider system load
    if 0.45 < p_slow < 0.55:
        if queue_size > 10:      # High load → Fast Path (boost throughput)
            return 0
        elif queue_size < 3:     # Low load → Slow Path (boost quality)
            return 1
        else:                    # Medium load → Use model prediction
            return model_label

    # 3. Non-grey Zone → Use model prediction
    return model_label
```

**Decision Tree**:
```
                   Router Prediction
                          ↓
                ┌─────────┴─────────┐
           High Confidence      Low Confidence
           (margin > 0.3)        (grey zone)
                ↓                     ↓
          Use Prediction         Check Queue
                              ┌──────┼──────┐
                         Queue>10  3≤Q≤10  Queue<3
                              ↓       ↓       ↓
                          Fast    Model   Slow
                          Path    Label   Path
```

**Rationale**:
- **High confidence**: Model is certain → trust the prediction
- **Grey zone + High load**: Prioritize Fast Path to prevent system congestion
- **Grey zone + Low load**: Have capacity for Slow Path → boost output quality
- **Load-Awareness**: Dynamically balances system load and service quality

---

## Performance Metrics

### 1. Classification Performance

**Dataset**: databricks/databricks-dolly-15k (6,224 samples)
- Fast Path (Label 0): `classification`, `summarization` (3,324 samples)
- Slow Path (Label 1): `creative_writing`, `general_qa` (2,900 samples)

**Train/Test Split**: 80/20 (4,979 train, 1,245 test)

**Best Model**: MiniLM-L6-v2 + instruction_sep_context

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **94.14%** |
| **F1 Score (macro)** | **0.941** |
| **F1 Score (weighted)** | **0.941** |
| **F1 Score (binary, pos_label=1)** | **0.937** |
| **Grey Zone Rate** | **1.77%** |

**Confusion Matrix**:
```
                Predicted
              Fast    Slow
Actual Fast    632     33
       Slow     40    540
```

**Model Comparison** (Ablation Study):

| Model | Text Format | Test Acc | F1 (macro) | Grey Zone |
|-------|-------------|----------|------------|-----------|
| MiniLM-L6-v2 | instruction_only | 86.59% | 0.866 | 4.58% |
| **MiniLM-L6-v2** | **instruction_sep_context** | **94.14%** | **0.941** | **1.77%** |
| mpnet-base-v2 | instruction_only | 87.71% | 0.877 | 3.13% |
| mpnet-base-v2 | instruction_sep_context | 93.82% | 0.938 | 2.73% |

**Key Findings**:
- Adding context improves accuracy by ~7-8%
- Lower grey zone rate means more confident predictions
- MiniLM-L6-v2 achieves best performance while being lightweight (80MB vs 420MB)

### 2. Router Latency & Throughput

**Benchmark Command**:
```bash
python3 benchmarks/benchmark_cache_miss.py
```

**Test Configuration**:
- 200 unique queries (cache miss scenario)
- 100 repeated queries (cache hit scenario)
- Model: MiniLM-L6-v2 + instruction_sep_context
- Batch size: 20 concurrent requests

**Router Latency Results** (x-router-latency header):

#### Cache Miss Performance (Unique Queries - Requires Inference)
| Metric | Latency |
|--------|---------|
| **P50** | 103.34ms |
| **P95** | 170.45ms |
| **P99** | 174.12ms |
| **Mean** | 110.54ms |
| **Min** | 92.74ms |
| **Max** | 176.30ms |

**Successful**: 200/200 requests
**Total Time**: 30.52s
**Throughput**: ~6.5 req/s (limited by sequential batches)

#### Cache Hit Performance (Repeated Queries - Direct Cache Return)
| Metric | Latency |
|--------|---------|
| **P50** | 0.01ms |
| **Mean** | 0.01ms |
| **Min** | 0.01ms |
| **Max** | 0.01ms |

**Performance Improvement**: Cache hits are **~10,000x faster** than cache misses.

**High Concurrency Test** (from `benchmarks/benchmark_quick.py`):
```bash
python3 benchmarks/benchmark_quick.py
```

| Test | Requests | Concurrency | P50 | P95 | P99 | Throughput |
|------|----------|-------------|-----|-----|-----|------------|
| Low | 100 | 10 | 0.01ms | 0.03ms | 0.05ms | 6.55 req/s |
| Medium | 500 | 50 | 0.01ms | 0.02ms | 0.03ms | 32.74 req/s |
| High | 1000 | 100 | 0.01ms | 0.01ms | 0.03ms | 63.83 req/s |

**Notes**:
- Latency measurements exclude Fast/Slow Path simulation time
- Cache hit rate dramatically affects overall performance (0.01ms vs 100ms+)
- Dynamic batching provides 10-20x throughput improvement over sequential processing
- In production with 20-30% cache hit rate, expected P50: ~70-80ms, P95: ~140-160ms

### 3. Load Testing

**Tool**: `locust` or `hey`

Example load test command:
```bash
# Install hey
brew install hey

# Run load test
hey -n 10000 -c 100 -m POST \
  -H "Content-Type: application/json" \
  -d '{"text":"How do I start running?"}' \
  http://localhost:8000/v1/query-process
```

---

## API Specification

### Endpoint

```
POST /v1/query-process
```

### Request

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "text": "User input string..."
}
```

### Response

**Status**: `200 OK`

**Headers**:
```
x-router-latency: 45.23
```
(Unit: milliseconds, router decision time only)

**Body**:
```json
{
  "label": "0"
}
```
or
```json
{
  "label": "1"
}
```

- `"0"`: Fast Path
- `"1"`: Slow Path

### Example

```bash
curl -X POST http://localhost:8000/v1/query-process \
  -H "Content-Type: application/json" \
  -d '{"text":"How do I start running?"}'
```

**Response**:
```json
{"label": "1"}
```

**Headers**:
```
x-router-latency: 48.56
```

---
## Development Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd query-gateway

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and process data
python data_download.py
```

### Training the Router (Required)

Models are not included in the repository due to size constraints. You need to train them locally:

```bash
# Train with best configuration
python train_router.py \
  --data dolly_processed.jsonl \
  --model all-MiniLM-L6-v2 \
  --text-format instruction_sep_context \
  --save-dir models

# Run ablation experiments
python train_router.py --ablation --data dolly_processed.jsonl

# Full ablation (includes mpnet)
python train_router.py --ablation-heavy --data dolly_processed.jsonl
```

### Running the Service

```bash
# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The service will automatically load the best model from `models/MiniLM_L6_v2___instruction_sep_context/`.

### Testing

```bash
# Smoke test
python test_router.py --model-dir models

# API test
curl -X POST http://127.0.0.1:8000/v1/query-process \
  -H "Content-Type: application/json" \
  -d '{"text":"How do I start running?"}'

# Health check
curl http://127.0.0.1:8000/health
```

### Docker (Coming Soon)

```bash
# Build image
docker build -t query-gateway:latest .

# Run container
docker run -p 8000:8000 query-gateway:latest
```

---

## System Design Details

### 1. Why Embedding-based Classifier?

**Considered Approaches**:

1. **TF-IDF + Logistic Regression**
   - ✅ Fast inference (matrix multiplication)
   - ✅ Small model size
   - ❌ Weak semantic understanding (spelling sensitive)
   - ❌ Based on keyword hard rules (substring matching)

2. **DistilBERT Fine-tuning**
   - ✅ Strongest semantic understanding
   - ✅ Direct text → label
   - ❌ Slow training and inference
   - ❌ Large model size (poor deployment)
   - ❌ Overkill for 4 categories → 2 paths

3. **Sentence-BERT + Logistic Regression** ✅ (Selected)
   - ✅ Strong semantic generalization
   - ✅ Off-the-shelf embeddings, easy to use
   - ✅ Efficient batch encoding (10-20x faster)
   - ✅ Probability outputs are interpretable
   - ✅ Only 6,224 samples, sufficient for training

### 2. Batching Window Trade-off

**Why 50ms window?**

| Window Size | Latency | Throughput | Trade-off |
|-------------|---------|------------|-----------|
| 10ms | Low (~15ms) | Poor (small batches) | Responsive but inefficient |
| **50ms** | **Medium (~45ms)** | **Good (avg 15-20 req/batch)** | **Balanced** ✅ |
| 100ms | High (~95ms) | Best (large batches) | High throughput but slow |

**Decision**: 50ms ensures P95 < 100ms while maximizing batch efficiency.

### 3. LRU Cache Design

**Why LRU?**
- Real-world queries often repeat (FAQs, common questions)
- Python `OrderedDict` natively supports O(1) LRU operations
- Automatic eviction prevents memory bloat

**Alternative**: Redis (for distributed systems)
- Current: In-memory OrderedDict (sufficient for single-instance)
- Future: Redis for multi-instance deployments

### 4. Load-Aware Decision Policy

**Scenario Examples**:

| Queue Size | p_slow | margin | Decision | Reason |
|------------|--------|--------|----------|--------|
| 2 | 0.95 | 0.90 | Slow (1) | High confidence, low load → trust model |
| 15 | 0.48 | 0.04 | Fast (0) | Grey zone + high load → protect system |
| 1 | 0.52 | 0.04 | Slow (1) | Grey zone + low load → boost quality |
| 8 | 0.15 | 0.70 | Fast (0) | High confidence → trust model |

This ensures the system dynamically adapts to both model confidence and system load.

---
## Project Structure

```
.
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI app with queue-based decision policy
│   ├── main2.py             # FastAPI app with dynamic threshold policy
│   └── service.py           # Model loading & batch inference layer
│
├── benchmarks/              # Performance benchmarking scripts
│   ├── benchmark_cache_miss.py   # Cache hit/miss performance test
│   ├── benchmark_quick.py        # High concurrency test
│   ├── benchmark.py              # Full end-to-end latency test
│   ├── benchmark_output.txt      # Quick benchmark results
│   ├── benchmark_results.json    # Full benchmark results
│   └── cache_benchmark.json      # Cache test results
│
├── models/                  # Trained models (not in repo, train locally)
│   ├── MiniLM_L6_v2___instruction_only/
│   ├── MiniLM_L6_v2___instruction_sep_context/  # Best model (94.14% acc)
│   ├── mpnet_base_v2___instruction_only/
│   └── mpnet_base_v2___instruction_sep_context/
│
├── train_router.py          # Training script with ablation experiments
├── test_router.py           # Smoke test for trained models
├── data_download.py         # Download & preprocess Dolly dataset
├── data_check.py            # Verify processed data integrity
│
├── dolly_processed.jsonl    # Processed dataset (not in repo, run data_download.py)
├── ablation_results.json    # Ablation experiment results (included)
│
├── requirements.txt         # Python dependencies
├── Dockerfile               # Multi-stage Docker build
├── .dockerignore            # Docker build exclusions
├── .gitignore               # Git exclusions
│
├── README.md                # Complete documentation
└── Note.md                  # Development notes & design decisions
```

### File Descriptions

#### Core Application

| File | Purpose | Key Features |
|------|---------|--------------|
| `app/main.py` | Main FastAPI application | Queue-based decision policy, batching, caching |
| `app/main2.py` | Alternative FastAPI app | Dynamic threshold policy with Semaphore |
| `app/service.py` | Service layer | Model loading, batch inference wrapper |

#### Training & Testing

| File | Purpose | Usage |
|------|---------|-------|
| `train_router.py` | Train semantic router | `python train_router.py --ablation --data dolly_processed.jsonl` |
| `test_router.py` | Smoke test models | `python test_router.py --model-dir models` |
| `data_download.py` | Download & process data | `python data_download.py` |
| `data_check.py` | Verify data integrity | `python data_check.py` |

#### Benchmarks

| File | Purpose | Usage |
|------|---------|-------|
| `benchmarks/benchmark_cache_miss.py` | Cache hit/miss performance analysis | `python3 benchmarks/benchmark_cache_miss.py` |
| `benchmarks/benchmark_quick.py` | High concurrency test (3 levels) | `python3 benchmarks/benchmark_quick.py` |
| `benchmarks/benchmark.py` | Full end-to-end latency test | `python3 benchmarks/benchmark.py` |
| `benchmarks/*.json` | Benchmark results in JSON format | For analysis and plotting |
| `benchmarks/*.txt` | Benchmark output logs | Human-readable results |

#### Models

**Note**: Models are not included in the repository. Train them locally using `train_router.py`.

All models are stored in `models/` with the naming convention:
```
{encoder}___{text_format}/
  ├── router.pkl         # Serialized router (classifier + metadata)
  └── metrics.json       # Training metrics (accuracy, F1, confusion matrix)
```

**Best Model**: `MiniLM_L6_v2___instruction_sep_context`
- Test Accuracy: 94.14%
- F1 Score (macro): 0.941
- Grey Zone Rate: 1.77%

#### Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies (FastAPI, scikit-learn, sentence-transformers, etc.) |
| `Dockerfile` | Multi-stage Docker build with health check |
| `.dockerignore` | Exclude unnecessary files from Docker image |
| `.gitignore` | Exclude temporary files, caches, logs from Git |

#### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Complete system documentation, API spec, deployment guide |
| `Note.md` | Development notes: design decisions, trade-off analysis |
| `ablation_results.json` | Detailed comparison of 4 model configurations |

---

## Future Improvements

1. **Distributed Caching**: Integrate Redis for multi-instance deployments
2. **A/B Testing**: Compare different decision policy parameters
3. **Monitoring**: Add Prometheus metrics and Grafana dashboards
4. **Auto-scaling**: Adjust batch window and max batch size based on load
5. **Model Update**: Hot-swap model without service restart

---

## License

MIT License

---

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with using FastAPI, Sentence Transformers, and asyncio**
