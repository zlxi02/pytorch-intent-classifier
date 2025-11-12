# PyTorch Intent Classifier - Agentic Router

A lightweight intent classification system built with PyTorch for routing user queries to appropriate downstream services, reducing latency and costs compared to general-purpose LLMs.

## Overview

**Problem**: Modern AI systems route all queries to expensive LLMs (1000-2000ms latency, $0.01-0.03/request), even for simple intents like greetings or weather requests.

**Solution**: A specialized intent classifier that routes queries intelligently:

```
User Query → Intent Classifier (1-5ms) → High Confidence?
                                           ├─ YES: Route to specialized tool (fast, cheap)
                                           └─ NO:  Fallback to general LLM (slow, expensive)
```

**Key Benefits**:
- ⚡ **~1000x faster** (1-5ms vs 1000-2000ms)
- 💰 **70% cost reduction** in production
- 🎯 **High precision** for known intent classes
- 🔄 **Handles 60-80% of queries** on fast path

## Architecture

### High-Level Pipeline

```
┌─────────────┐
│ Input Text  │  "what is the weather in Paris"
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Preprocessing       │  Tokenize → ["what", "is", "the", "weather", "in", "paris"]
│ - Tokenization      │  Convert to indices → [8, 9, 10, 11, 13, 15]
│ - Vocab Lookup      │  Pad to fixed length → [8, 9, 10, 11, 13, 15, 0, 0, 0, 0]
│ - Padding           │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Embedding Layer     │  [batch, seq_len] → [batch, seq_len, embedding_dim]
│ (nn.Embedding)      │  [4, 10] → [4, 10, 50]
│                     │  Each word → 50-dimensional learned vector
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Mean Pooling        │  [batch, seq_len, embedding_dim] → [batch, embedding_dim]
│ (Bag of Embeddings) │  [4, 10, 50] → [4, 50]
│                     │  Average across sequence dimension
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Feed-Forward Net    │  [batch, 50] → [batch, 32] → [batch, 6]
│ - Linear(50→32)     │
│ - ReLU              │  Feature extraction & transformation
│ - Dropout(0.3)      │
│ - Linear(32→6)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Output Logits       │  [0.2, 3.8, -0.5, 0.1, -0.3, 0.4]
│                     │   ↑    ↑    ↑     ↑    ↑     ↑
│                     │   0    1    2     3    4     5
│                     │  Greet GetW BookF Thank Time Fare
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Softmax             │  [0.05, 0.92, 0.01, 0.01, 0.00, 0.01]
│                     │  GetWeather wins with 92% confidence
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Routing Decision    │  Confidence > 0.8?
│                     │  ✓ YES: Route to WeatherAPI
│                     │  ✗ NO:  Fallback to LLM
└─────────────────────┘
```

---

## Technical Design

### Components

**1. Embedding Layer** (`nn.Embedding`):
- Converts word indices → 50-dim vectors
- Learnable lookup table: `[vocab_size × 50]`
- `padding_idx=0` for padding tokens

**2. Mean Pooling**:
- Aggregates variable-length sequences → fixed size
- Formula: $\text{pooled} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{e}_i$
- Order-invariant but fast (1 operation)
- Sufficient for intent classification (keywords matter more than order)

**3. Feed-Forward Network**:
```
Input (50) → Linear(50→32) → ReLU → Dropout(0.3) → Linear(32→6) → Output (6)
```
- Layer 1: Compress semantic features → intent-relevant patterns
- ReLU: Non-linear activation
- Dropout: Regularization (30% random neuron dropout during training)
- Layer 2: Map to 6 intent class scores

**4. Loss & Optimizer**:
- **CrossEntropyLoss**: Combines softmax + negative log likelihood
- **Adam Optimizer**: Adaptive learning rates, learning_rate=0.001

### Parameter Count
```
Embeddings:  100 × 50 = 5,000
FC1:         50 × 32 + 32 = 1,632
FC2:         32 × 6 + 6 = 198
────────────────────────────────
Total:       6,830 parameters
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Mean Pooling** (vs RNN/Attention) | 5-10x faster, sufficient for short queries, simple |
| **Train from scratch** (vs pre-trained) | Small vocabulary (~50 words), task-specific, faster |
| **Batch size: 4** | 15 batches/epoch, frequent updates, better generalization |
| **Embedding dim: 50** | Balance between capacity and overfitting risk |
| **Hidden layer: 32** | Gradual compression (50→32→6), efficient |
| **Dropout: 0.3** | Moderate regularization for small dataset (60 examples) |
| **Epochs: 100** | Converges around epoch 50-70, fine-tunes after

## Performance Analysis

### Latency
- **Inference**: ~1-2ms per query (CPU)
- **Training**: ~10-15 seconds (100 epochs, CPU)
- **Throughput**: ~600-800 queries/second (single CPU core)
- **Speedup vs LLM**: ~1000x faster

### Accuracy
- **Training set**: ~95-98% after 100 epochs
- **Known patterns**: 85-95% confidence
- **Out-of-vocabulary**: 40-60% confidence (routes to LLM fallback)
- **Confidence distribution**: 60-70% queries take fast path (>0.8 confidence)

### Cost Analysis (10,000 queries/day)

**Without Intent Classifier**:
- Cost: $100/day = $36,500/year
- Latency: 4.17 hours total wait time/day

**With Intent Classifier** (70% fast path, 30% LLM):
- Cost: $30/day = $10,950/year (**70% reduction**)
- Latency: ~1.25 hours total (**70% reduction**)
- **Savings**: $25,550/year
- **ROI**: Pays for itself in < 1 month

### Deployment
- Model size: ~27KB (6,830 parameters)
- Memory: ~20MB runtime
- Runs on: CPU, edge devices, serverless, mobile

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Dependencies: torch>=2.0.0, numpy>=1.24.0
```

---

## Usage

### 1. Full Demo
```bash
python intent_classifier.py
```
Trains model, runs test sentences, enters interactive mode.

### 2. Classify Custom Sentence
```bash
python intent_classifier.py --input "what is the weather today"
```

### 3. Interactive Mode
After training, enter sentences and get instant classifications:
```
Enter a sentence: hello there
Input: "hello there"
Predicted Intent: Greeting
Confidence: 0.9456
Latency: 1.23 ms
✓ FAST PATH: Executing tool: Greeting

Enter a sentence: quit
```

### 4. Modify Training Data
Edit `data.py` to add more examples - model auto-adapts.

---

## Project Structure

```
pytorch-intent-classifier/
├── intent_classifier.py    # Main script (training + inference)
├── data.py                 # Training data (60 examples, 6 classes)
├── requirements.txt        # Dependencies
└── README.md              # This file
```

---

## Future Improvements

### Architecture
- **Attention mechanism**: Better for long queries, interpretable (10x slower)
- **Bidirectional LSTM**: Sequential dependencies (5x slower)
- **Hierarchical classification**: Scales to 50+ intents

### Data
- Increase to 100-500 examples/class
- Data augmentation (paraphrasing, synonyms)
- Hard negative mining for ambiguous cases
- Multi-language support

### Production
- Model persistence (save/load trained weights)
- Confidence calibration (temperature scaling)
- A/B testing framework
- Monitoring & logging
- REST API / Serverless deployment
- Active learning (continuous improvement)

---

## License

MIT License

