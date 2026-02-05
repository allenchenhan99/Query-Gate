## AI 流量管制中心
Data: 2026/02/05 (5hrs)
### Machine Learning System Design
Focus on **System Engineering** and the balance of **Latency / Throughput**

大致上流程可先拆兩部分：一是語意分類訓練，二是後端server建立和router implemenetation

---
# 語意分類器 Semantic Router
以 [databricks](https://huggingface.co/datasets/databricks/databricks-dolly-15k)料集作為設計依據，把 **prompt** label 成 0 / 1
只需看四種類別，總共 6224 筆：
- **Fast Path (Label 0)**: `classification`, `summarization`
- **Slow Path (Label 1)**: `create_writing`, `general_qa`
---
## Possible Solution
### 1. Embedding + LogisticRegression
>Sentence-BERT + LogisticRegression
  泛化能力佳，現成可用，6224筆對於來訓練負擔不大
  分類的邊界也比較清楚（因為是4 categories into 2 paths）
  Encoder 支援 batch encoding
  Logistic Regression出來的predict probability有解釋意義
- 
### 2. TF-IDF Vectorizer + Logistic Regression

>直覺上也會想到TFIDF（Word2Vec）
>推論速度極快（畢竟只是矩陣乘法）
>模型小依賴小好部署到Docker
>訓練也理所當然快
>解釋性上，可以針對哪個字關鍵字權重
>因為字的拼寫敏感，所以語意理解會比較弱
>但重點是他是based on **key word hard rule**，所以某方面來講可能也是直接用string match的方式

### 3. Distillation(DistilBERT fine-tuning)
>語意理解會最強準確度會最高
>直接text $\rightarrow$ label，能夠理解前後文
>但缺點是訓練久模型大，部署上比較沒優勢
>目標只有4 labels to 2 paths，有點殺雞用牛刀

最後選擇**Embedding + LR**來當作前半段的語意分類器

## Ablation Experiments
選擇Embedding + LR後，因為有考慮到其輸出的$P(y = 1|x)$在統計上有真實的信心度，比起**Naive Bayes**，雖然還是一樣輸出機率，但因為假設特徵獨立，所以output出來往往都會比較極端。而至於**SVM (Support Vector Machine)**，雖然可以使用platt-scaling轉成機率，但是考量到需再透過一步程序轉機率，且計算上也比較有負擔，因此選擇Logistic Regression。

後續還有特地做 **2x2 的消融實驗 (Ablation Study)**
在 `train_router.py` 中做以下四種配置：

| Experiment ID | Model Backbone  | Input Format              | Hypothesis (假設)                                              |
| :------------ | :-------------- | :------------------------ | :----------------------------------------------------------- |
| **Exp 1**     | `MiniLM-L6-v2`  | `instruction_only`        | 最輕量、速度最快，測試僅靠指令是否足夠分類。                                       |
| **Exp 2**     | `MiniLM-L6-v2`  | `instruction_sep_context` | **(Selected)** 增加 Context 資訊，預期能提升模糊指令的辨識度，且維持 MiniLM 的速度優勢。 |
| **Exp 3**     | `mpnet-base-v2` | `instruction_only`        | 使用更強大的模型架構，測試模型能力是否能彌補資訊量的不足。                                |
| **Exp 4**     | `mpnet-base-v2` | `instruction_sep_context` | 理論上的效能天花板 (Upper Bound)，但推論成本最高。                             |

#### Experimental Insights
1. **Context 的重要性**： 實驗發現，包含 `Context` 的組別 (Exp 2, Exp 4) 在 F1-Score 上普遍優於僅有 Instruction 的組別。這證實了在判斷 `Summarization` vs `Creative Writing` 時，原始文本的長度與特徵是關鍵變數。
2. **Backbone 的權衡 (Speed vs Accuracy)**： 雖然 `mpnet-base-v2` (Exp 4) 的準確度略高於 `MiniLM`，但其參數量 (110M) 是 `MiniLM` (22M) 的 5 倍，導致推論延遲顯著增加。
3. **Final Decision**： 考慮到 Gateway 的高併發需求，選擇了 **Exp 2 (`MiniLM-L6-v2` + `instruction_sep_context`)**。

#### Ablation Results

| Experiment | Model Backbone  | Text Format              | Test Acc | F1 (macro) | F1 (binary) | Grey Zone Rate |
|:-----------|:----------------|:-------------------------|:---------|:-----------|:------------|:---------------|
| **Exp 1**  | `MiniLM-L6-v2`  | `instruction_only`       | 86.59%   | 0.866      | 0.859       | 4.58%          |
| **Exp 2**  | `MiniLM-L6-v2`  | `instruction_sep_context`| **94.14%** | **0.941** | **0.937**   | **1.77%**      |
| **Exp 3**  | `mpnet-base-v2` | `instruction_only`       | 87.71%   | 0.877      | 0.872       | 3.13%          |
| **Exp 4**  | `mpnet-base-v2` | `instruction_sep_context`| 93.82%   | 0.938      | 0.933       | 2.73%          |

**Key Findings:**
- **Context 帶來 7-8% 的準確度提升** (Exp1→Exp2: +7.55%, Exp3→Exp4: +6.11%)
- **MiniLM-L6-v2 達到最佳性能** 同時保持輕量級 (80MB vs 420MB)
- **Grey Zone Rate 降低 60%** (從 4.58% → 1.77%)，代表模型預測能更有信心
- **F1 Binary Score 提升** 表示 Slow Path (Label 1) 的分類能更精準

---
## Backend System Architecture
需滿足：
1. **High Concurrency**: 處理高併發請求
2. **Asynchronous I/O**: 非同步設計避免阻塞
3. **Dynamic Batching**: 短時間視窗內聚合多筆請求
4. **Caching**: 減少重複計算
5. **Decision Logic**: 權衡 Cost / Confidence / Load

survey 一輪後發現 **FastAPI** 支援 async/await
高效能非同步server: **Uvicorn**

架構流程可以理解為下：
HTTP Request → Check Cache (**Hit**的話就Return)
(**Miss**) → Enqueue to Batch Queue
→ Background Batch Worker
→ Batch Inference **(1次推論N個請求)**
→ Route to Fast/Slow Path
→ Return + x-router-latency header
              
---
## Dynamic Batching
因為禁止對每一個 HTTP Request 單獨進行 Inference
實作方式：**Time-based Windowing** + **Adaptive Collection**
##### 配置參數
BATCH_WINDOW_MS = 50    # 批次收集時間窗口（毫秒）
MAX_BATCH_SIZE = 32     # 最大批次大小

**機制流程** 
  1. Blocking Wait    → 等第一個 request 進 Queue（確保有工作）
  2. Start Window     → 設定 deadline = now + 50ms
  3. Fast Collection  → get_nowait() 非阻塞快速拉取更多 requests
  4. Trigger Condition→ 達到以下任一條件就觸發推論：
                        • Batch Size = 32 (MAX_BATCH_SIZE)
                        • Time = 50ms (deadline 到期)
  5. Batch Inference  → encoder.encode(batch_texts) 一次推論所有
  6. Distribution     → 結果寫回各自的 Future

---
## Decision Logic
根據作業要求，Decision Logic 需要綜合考量三個因素： 
- **Cost-Weighting**: Fast Path 成本低，Slow Path 成本高 
- **Confidence**: Router 的預測信心度 
- **Load-Awareness**: 系統當前負載狀況
### main.py
先針對預測出來的 Slow Path 的機率，離 0.5 越遠(margin)代表越有信心，因此可直接回傳label
而threshold以0.3為界線，若預測出來之Slow Path機率為 0.45 ~ 0.55，則先認定為模糊地帶（Grey_Zone）
針對Grey zone的candidates，分成兩種負載情況
1. 高負載 (queue_size > 10)
   ```python
   if queue_size > 10:
		return 0  # 強制 Fast Path
   ```
   理由為：系統壓力大，需要快速處理請求，因此可以犧牲一些品質換取throughput。而以10為界線的原因則是根據`BATCH_WINDOW_MS = 50ms` 和 `MAX_BATCH_SIZE = 32`來預估的。如果 queue > 10，意味著積壓了 ~300ms 的請求，這時系統已經開始出現延遲，需要優先處理。系統閒置，有餘力處理複雜請求。
2. 低負載 (queue_size < 3)
   ```python
	elif queue_size < 3:
	    return 1  # 強制 Slow Path
   ```
3. 中等負載 (3 ≤ queue_size ≤ 10)
   ```python
   else: 
	   return model_label
```
	負載正常，相信模型的原始預測，不需要過度干預。

### main2.py
在 V1 版本中，我使用 `queue_size` 來判斷系統負載，但發現這個指標有侷限性：
- Queue Size 只反映「等待推論的請求數」
- 無法反映**正在執行 Slow Path 的請求數**
- Cache Hit 時會忽略當前負載狀況
因此在 V2 中，我改用 **Semaphore 追蹤實際負載** + **動態閾值調整**，讓決策更精準地反映系統狀態。

#### 核心改進
##### 1. 從 Queue Size 到 Semaphore
**main 的做法:**
```python
queue_size = batch_queue.qsize()  # 只看排隊數量
```

**問題:**
- Queue 只記錄**等待 batching 的請求**
- 不知道有多少 Slow Path 正在執行

**main2 的改進:**
```python
# 使用 Semaphore 限制並追蹤 Slow Path 併發數
SLOW_PATH_SEMAPHORE = asyncio.Semaphore(MAX_SLOW_CONCURRENCY)

def get_current_slow_load() -> int:
    """獲取當前 Slow Path 活躍數量"""
    return MAX_SLOW_CONCURRENCY - SLOW_PATH_SEMAPHORE._value

# 在執行 Slow Path 時自動管理
async with SLOW_PATH_SEMAPHORE:
    sleep_time = random.uniform(1.0, 3.0)
    await asyncio.sleep(sleep_time)
    # 離開 context 後自動釋放，active count 減 1
```

優勢在精確追蹤「正在執行」的 Slow Path 數量、自動管理併發限制（超過 50 個會自動等待）以及線程安全，適合高併發場景。

##### 2. 動態閾值（二次方懲罰）
**main1 的做法（固定閾值）:**
- 閾值是硬編碼的 (0.45, 0.55)
- 只有在「灰區」才考慮負載
- 負載影響是「階梯式」的（3 vs 10），不夠平滑

**main2 使用動態計算:**
```python
# 計算負載率
load_ratio = current_active / MAX_SLOW_CONCURRENCY  # 0.0 ~ 1.0

# 動態閾值（二次方懲罰）
dynamic_threshold = BASE_THRESHOLD + ADAPTIVE_ALPHA * (load_ratio ** 2)
# 例: 0.5 + 0.45 * (0.8)^2 = 0.788

# 判定
if p_slow > dynamic_threshold:
    return 1, "slow_granted"
else:
    return 0, "load_shedding"
```

**Example:**
```
負載率 = 0%   → 閾值 = 0.50 (寬鬆)
負載率 = 20%  → 閾值 = 0.52 (微調)
負載率 = 50%  → 閾值 = 0.61 (明顯)
負載率 = 80%  → 閾值 = 0.79 (嚴格)
負載率 = 100% → 閾值 = 0.95 (極嚴)
```

- **低負載時**: 閾值緩慢上升 → 不過度干預
- **高負載時**: 閾值快速上升 → 積極限流
- **平滑過渡**: 避免「突然切換」導致的抖動

##### 3. Cache 重新決策
**main 的做法:**
```python
# Cache 存最終決策
set_cache(text, final_label, p_slow, margin)

# Cache Hit 時直接用
if cached_result:
    label, _, _ = cached_result
    return label  # 舊決策，忽略當前負載
```

**問題:**
- Cache Hit 時無法感知當前負載
- 同樣的 query，低負載時走 Slow，高負載時應該降級但做不到

**main2 的改進:**
```python
# Cache 只存模型輸出
set_cache(text, model_label, p_slow, margin)

# Cache Hit 時重新決策
if cached_result:
    model_label, p_slow, margin = cached_result
    final_label, reason, threshold = apply_decision_policy(
        model_label, p_slow, margin  # 根據當前負載重新算
    )
```
優勢在 Cache Hit 也能動態適應負載，省去 embedding encode 時間，但同時保留決策靈活性
## 配置參數
```python
BASE_THRESHOLD = 0.5         # 基礎門檻（無負載時）
ADAPTIVE_ALPHA = 0.45        # 負載敏感度（控制曲線陡峭度）
MAX_SLOW_CONCURRENCY = 50    # Slow Path 最大併發數
HIGH_CONFIDENCE_MARGIN = 0.3 # 高信心閾值
```
##### ALPHA = 0.452的用意
這讓滿載時 `T_max = 0.5 + 0.45 = 0.95`：
- 滿載時需要 95% 的機率才能走 Slow
- 基本上只有「極度確定」才會通過
- 保護系統不會崩潰
最終也能在**Cost / Confidence / Load** 三者的動態達到相較完美平衡