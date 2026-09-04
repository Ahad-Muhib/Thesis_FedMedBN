# FedMed-BN: Complete Implementation Roadmap
## Privacy-Preserving Federated Learning for Bangla Medical Text

> **Your situation:** Starting from scratch, Google Colab, no datasets downloaded, 1-2 months to final defence.
> **Strategy:** Move fast. Build in phases. Get a working end-to-end pipeline first, then improve.

---

## 🔑 THE BIG PICTURE — What You're Actually Building

You are building a **simulation** of a federated learning system where:
- **3 simulated hospitals** (not 5 — reduced for speed & Colab memory) each have their own slice of Bangla medical text data
- Each hospital trains a **BanglaBERT** model locally on its own data
- No hospital shares its raw data — only model weight updates
- A **central server** aggregates these updates using **FedAvg**
- **Differential Privacy** (gradient clipping + noise) protects against gradient leakage attacks
- The task is **Medical Named Entity Recognition (NER)** — detecting diseases, medicines, organs etc. in Bangla text

### What Makes This Different From the Chinese Paper (FLCMC)?
| Aspect | Chinese Paper (FLCMC) | Your Project (FedMed-BN) |
|--------|----------------------|--------------------------|
| Language | Chinese | Bangla |
| Task | Text Classification (symptom detection) | Named Entity Recognition (NER) |
| Model | LSTM + Word2Vec | BanglaBERT (Transformer) |
| FL Algorithm | FedPA/FedPAP (custom attention aggregation) | FedAvg + Differential Privacy |
| Privacy | Only architectural (no formal DP) | Formal DP with Opacus (ε, δ guarantees) |
| Clients | 10-30 simulated | 3 simulated hospitals |
| Framework | TensorFlow | PyTorch + Flower + Opacus |

---

## 📅 TIMELINE: 6-Week Sprint Plan

### Week 1: Setup + Data (Days 1-7)
### Week 2: Centralized Baseline (Days 8-14)
### Week 3: Federated Learning Pipeline (Days 15-21)
### Week 4: Differential Privacy Integration (Days 22-28)
### Week 5: Experiments + Results (Days 29-35)
### Week 6: Thesis Writing + Defence Prep (Days 36-42)

---

## 📋 PHASE 1: Environment Setup + Data Preparation (Week 1)

### Step 1.1 — Set Up Google Colab Environment

Create a notebook called `01_setup_and_data.ipynb`:

```python
# Cell 1: Install all required libraries
!pip install transformers datasets torch flower-federated opacus
!pip install bnlp_toolkit indic-nlp-library seqeval
!pip install scikit-learn pandas matplotlib seaborn
```

```python
# Cell 2: Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
# You NEED GPU runtime. Go to Runtime > Change runtime type > T4 GPU
```

### Step 1.2 — Download the 3 Datasets

**Dataset 1: Bangla MedER (Primary — 6,895 records)**
- Search on Kaggle: "Bangla MedER" by Tanjim Tahar Taurpa
- OR find on the source mentioned in your thesis (PLOS One, Sept 2025)
- This has disease/specialty labels — you'll use it for classification AND as base text

**Dataset 2: BanglaBioMed (818 sentences, 4 entity types)**
- Source: ACL BioNLP 2022 workshop
- Search: "BanglaBioMed NER dataset"
- Entity types: Anatomy, Drug, Disease, Procedure

**Dataset 3: Bangla-MedER (2,980 statements, 6 entity types)**
- Source: Mendeley Data (October 2025)
- Entity types: Medicine, Organ, Disease, Hormone, Pharmacological Class, Common Medical Terms
- This is your MAIN NER dataset

```python
# Cell 3: Download datasets
# Option A: If on Kaggle
# !kaggle datasets download -d <dataset-path>

# Option B: Manual upload to Colab
from google.colab import files
# uploaded = files.upload()  # Upload your downloaded dataset files

# Option C: Mount Google Drive (RECOMMENDED — persistent storage)
from google.colab import drive
drive.mount('/content/drive')
# Put datasets in /content/drive/MyDrive/FedMed-BN/data/
```

### Step 1.3 — Explore and Understand the Data

```python
# Cell 4: Load and explore each dataset
import pandas as pd

# Adjust paths based on actual file format (CSV, JSON, etc.)
# df_meder = pd.read_csv('/content/drive/MyDrive/FedMed-BN/data/bangla_meder.csv')
# print(f"Bangla MedER shape: {df_meder.shape}")
# print(df_meder.head())
# print(df_meder.columns.tolist())
# print(df_meder['label_column'].value_counts())  # Check class distribution
```

**What to look for:**
- How many columns? What are they?
- What format are NER annotations in? (BIO tags? Separate entity spans?)
- How many unique entity types?
- Any missing values?
- Sample some Bangla text — does it look clean?

### Step 1.4 — Preprocess and Create NER-Ready Data

```python
# Cell 5: Standardize into unified NER format
# Target format: each row = one sentence with token-level BIO labels
# Example:
# tokens:  ["আমার", "মাথা", "ব্যথা", "প্যারাসিটামল", "খাচ্ছি"]
# labels:  ["O",     "B-ORGAN", "B-DISEASE", "B-MEDICINE",    "O"]

# You need to convert your dataset into this format
# The exact code depends on the raw format of your downloaded data
```

### Step 1.5 — Create Non-IID Hospital Partitions

This is CRITICAL for federated learning simulation:

```python
# Cell 6: Partition data into 3 hospital clients (Non-IID)
import numpy as np

def create_non_iid_partitions(dataset, num_clients=3):
    """
    Create non-IID partitions simulating hospitals with different specializations.
    
    Strategy: Each hospital gets a SKEWED distribution of entity types.
    Hospital A: More disease-heavy samples (like a diagnostic center)
    Hospital B: More medicine-heavy samples (like a pharmacy/prescription clinic)
    Hospital C: Mixed but with organ/procedure focus (like a surgery department)
    """
    # Sort by dominant entity type, then distribute unevenly
    # Actual implementation depends on your data structure
    pass

# Split each partition: 80% train, 20% test
from sklearn.model_selection import train_test_split
```

**Why 3 hospitals instead of 5?**
- Colab has limited RAM (12-15 GB on free tier)
- BanglaBERT is ~440MB — loading 5 copies simultaneously is risky
- 3 is still enough to demonstrate federated learning
- Your thesis says 5 but you can justify 3 in your defence as a computational constraint

---

## 📋 PHASE 2: Centralized Baseline (Week 2)

**WHY:** You need a centralized (non-federated) result to compare against. This proves your federated approach works. Train BanglaBERT on ALL data combined, without any federation.

### Step 2.1 — Create the BanglaBERT NER Model

Create notebook `02_centralized_baseline.ipynb`:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load BanglaBERT
MODEL_NAME = "csebuetnlp/banglabert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# For NER, use AutoModelForTokenClassification
# num_labels = number of BIO tags (e.g., B-DISEASE, I-DISEASE, B-MEDICINE, ..., O)
num_labels = 13  # Adjust based on your actual tag count
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=num_labels
)
```

### Step 2.2 — Tokenize for NER

```python
# NER tokenization is tricky — subword tokens need aligned labels
def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        is_split_into_words=True  # Important for NER!
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignore padding/special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignore subword tokens
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized["labels"] = labels
    return tokenized
```

### Step 2.3 — Train Centralized Model

```python
from torch.utils.data import DataLoader
from torch.optim import AdamW
from seqeval.metrics import classification_report, f1_score

# Training loop
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
model.to('cuda')

EPOCHS = 5
BATCH_SIZE = 16

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**{k: v.to('cuda') for k, v in batch.items()})
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluate
    model.eval()
    # ... compute precision, recall, F1 per entity type
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

### Step 2.4 — Record Baseline Results

**Save these numbers! You'll compare federated results against them:**
- Overall Accuracy
- Per-entity Precision, Recall, F1
- Macro F1, Weighted F1
- Training time

---

## 📋 PHASE 3: Federated Learning Pipeline (Week 3)

This is the CORE of your project. Create notebook `03_federated_learning.ipynb`:

### Step 3.1 — Understand Flower Framework

Flower (flwr) has 3 key concepts:
1. **Client**: Trains locally, sends updates to server
2. **Server**: Aggregates updates from all clients
3. **Strategy**: How aggregation works (FedAvg in your case)

### Step 3.2 — Define the Flower Client

```python
import flwr as fl
from collections import OrderedDict

class BanglaBERTClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
    
    def get_parameters(self, config):
        """Return model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train on local data."""
        self.set_parameters(parameters)
        
        self.model.train()
        self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        # Local training for E epochs
        LOCAL_EPOCHS = 3
        for epoch in range(LOCAL_EPOCHS):
            for batch in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()})
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate on local test data."""
        self.set_parameters(parameters)
        self.model.eval()
        self.model.to(self.device)
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                outputs = self.model(**{k: v.to(self.device) for k, v in batch.items()})
                total_loss += outputs.loss.item()
                # Calculate accuracy (ignoring -100 labels)
                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"].to(self.device)
                mask = labels != -100
                correct += (predictions[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        
        accuracy = correct / total if total > 0 else 0
        return float(total_loss / len(self.test_loader)), len(self.test_loader.dataset), {"accuracy": accuracy}
```

### Step 3.3 — Simulation Setup (Colab-Compatible)

**IMPORTANT:** On Colab, you can't run a real distributed system. You use Flower's **simulation mode**:

```python
import flwr as fl
from flwr.simulation import start_simulation

def client_fn(cid: str):
    """Create a client for the given client ID."""
    client_id = int(cid)
    
    # Load the partition for this client
    train_loader = client_train_loaders[client_id]
    test_loader = client_test_loaders[client_id]
    
    # Each client gets a FRESH copy of the model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    
    return BanglaBERTClient(model, train_loader, test_loader, device='cuda')

# FedAvg Strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,        # Use ALL clients each round (we only have 3)
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

# Run simulation
NUM_ROUNDS = 10  # Start with 10, increase to 20 if time permits

history = start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_gpus": 1.0, "num_cpus": 2.0},
)
```

### Step 3.4 — Record Federated Results (Without DP)

Save these to compare with centralized AND with DP later:
- Per-round accuracy and loss (plot convergence curves)
- Final global model metrics
- Per-client metrics (shows non-IID effect)

---

## 📋 PHASE 4: Differential Privacy Integration (Week 4)

### Step 4.1 — Add Opacus to the Client

Modify the client's `fit()` method:

```python
from opacus import PrivacyEngine

class DPBanglaBERTClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, 
                 target_epsilon=3.0, target_delta=1e-5, 
                 max_grad_norm=1.0, noise_multiplier=1.1):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        self.model.to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        # Wrap with Opacus PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        
        LOCAL_EPOCHS = 3
        for epoch in range(LOCAL_EPOCHS):
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(**{k: v.to(self.device) for k, v in batch.items()})
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        # Get actual epsilon spent
        epsilon = privacy_engine.get_epsilon(delta=self.target_delta)
        print(f"Client epsilon after training: {epsilon:.2f}")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"epsilon": epsilon}
    
    # get_parameters, set_parameters, evaluate same as before
```

### Step 4.2 — Privacy-Utility Trade-off Experiments

Run experiments with DIFFERENT epsilon values:

```python
# Experiment grid
epsilon_configs = [
    {"noise_multiplier": 2.0, "label": "High Privacy (ε ≈ 1)"},
    {"noise_multiplier": 1.1, "label": "Medium Privacy (ε ≈ 3)"},
    {"noise_multiplier": 0.5, "label": "Low Privacy (ε ≈ 8)"},
]

results = {}
for config in epsilon_configs:
    # Run federated training with this noise level
    # Record: final accuracy, F1, actual epsilon
    pass
```

---

## 📋 PHASE 5: Experiments & Results (Week 5)

### Step 5.1 — Experiments to Run

You need THESE specific experiments for your thesis:

| Experiment | What It Shows | For Thesis Section |
|--|--|--|
| 1. Centralized BanglaBERT (all data combined) | Baseline performance | 5.1 |
| 2. Federated (3 clients, NO DP) | FL works for Bangla NER | 5.2 |
| 3. Federated + DP (ε ≈ 3.0) | Privacy with acceptable accuracy | 5.3 |
| 4. Privacy-Utility Trade-off (vary ε) | Shows ε vs accuracy curve | 5.4 |
| 5. Per-client analysis | Non-IID effect on each hospital | 5.5 |

### Step 5.2 — Graphs and Visualizations to Generate

Create notebook `04_results_visualization.ipynb`:

```
Graph 1: Training loss convergence (centralized vs federated vs federated+DP)
Graph 2: Per-round accuracy across communication rounds
Graph 3: Privacy budget (ε) vs. F1-score trade-off curve
Graph 4: Per-client accuracy comparison (Hospital A vs B vs C)
Graph 5: Confusion matrix for entity types
Graph 6: Per-entity-type F1 comparison (centralized vs federated vs DP)
Graph 7: Bar chart comparing all approaches side by side
```

### Step 5.3 — Results Table Template

| Method | Accuracy | Precision | Recall | F1 | ε (Privacy) |
|--------|----------|-----------|--------|-----|-------------|
| Centralized BanglaBERT | ? | ? | ? | ? | ∞ (no privacy) |
| FedAvg (3 clients, no DP) | ? | ? | ? | ? | ∞ (no formal DP) |
| FedAvg + DP (ε ≈ 1) | ? | ? | ? | ? | ~1.0 |
| FedAvg + DP (ε ≈ 3) | ? | ? | ? | ? | ~3.0 |
| FedAvg + DP (ε ≈ 8) | ? | ? | ? | ? | ~8.0 |

**Expected pattern:** Centralized > Federated (no DP) > Federated (ε≈8) > Federated (ε≈3) > Federated (ε≈1)

---

## 📋 PHASE 6: Thesis Completion + Defence Prep (Week 6)

### Step 6.1 — Fill Thesis Gaps

| Section | Status Now | What to Write |
|---------|-----------|---------------|
| 3.3 Algorithmic Flow | Empty | Add FedAvg pseudocode + DP-FedAvg algorithm |
| 5.1-5.6 Results | Empty | All experimental results, tables, graphs |
| 7.1 Standards | Empty | Map to BAETE standards |
| 7.2 Design Constraints | Empty | Hardware/software constraints |
| Appendix A: Source Code | Empty | Add key code snippets |

### Step 6.2 — Fix Known Issues
- **Chapter 7 copy-paste error**: "PID-based line following robot project" → replace with "FedMed-BN"
- **Section 4.5 broken reference**: Fix "Section ??" cross-reference
- Change "5 hospitals" to "3 hospitals" everywhere (or justify the reduction)

### Step 6.3 — Defence Preparation
- Prepare 15-20 slides
- Know your numbers cold (F1 scores, epsilon values)
- Anticipate questions:
  - "Why 3 hospitals not 5?" → Colab memory constraint, 3 still demonstrates FL
  - "Why FedAvg not FedProx?" → Simpler, proven baseline, focus is on DP not aggregation
  - "How does DP affect accuracy?" → Show your trade-off curve
  - "Is this publishable?" → Yes, first Bangla medical NER with FL+DP
  - "What's the real-world deployment path?" → Future work, needs real hospital partnerships

---

## 🛠️ PRACTICAL TIPS

### Google Colab Survival Guide
1. **Save to Drive frequently** — Colab disconnects after ~90 min idle
2. **Use checkpoints** — Save model after each FL round
3. **Colab Pro** ($10/mo) gives you more RAM and longer sessions — WORTH IT for this project
4. **Reduce batch size** if you get OOM errors (try 8 instead of 16)
5. **Use `torch.cuda.empty_cache()`** between clients

### If BanglaBERT Is Too Heavy for Colab
Fallback options (in order of preference):
1. **Freeze lower layers** — Only fine-tune top 3-4 layers instead of all 12
2. **Use `csebuetnlp/banglabert-small`** if available
3. **Use LoRA/PEFT** — Parameter-efficient fine-tuning (much less memory)
4. **Reduce max_length** from 128 to 64

### Opacus Compatibility Warning
Opacus has strict requirements:
- Must use `BatchNorm` → `GroupNorm` (BanglaBERT uses LayerNorm which is fine)
- DataLoader must have `drop_last=True` if batch sizes are uneven
- Some HuggingFace model internals may need patching — test early!

---

## 🎯 PRIORITY ACTIONS — Start TODAY

1. **Download the 3 datasets** — Find their exact URLs/Kaggle pages
2. **Create Google Colab notebook** with GPU runtime
3. **Install libraries and load BanglaBERT** — just verify it loads
4. **Load + explore ONE dataset** — understand its format
5. **Come back to me** with the dataset format and I'll write you the exact preprocessing code

---

## 📁 Suggested Project Structure (on Google Drive)

```
/FedMed-BN/
├── data/
│   ├── raw/                    # Original downloaded datasets
│   ├── processed/              # Cleaned, tokenized data
│   └── partitions/             # Non-IID hospital splits
├── notebooks/
│   ├── 01_setup_and_data.ipynb
│   ├── 02_centralized_baseline.ipynb
│   ├── 03_federated_learning.ipynb
│   ├── 04_dp_integration.ipynb
│   └── 05_results_visualization.ipynb
├── models/
│   ├── centralized/            # Saved centralized model
│   ├── federated/              # Saved FL model (no DP)
│   └── federated_dp/           # Saved FL+DP model
├── results/
│   ├── figures/                # Generated graphs
│   └── metrics/                # Saved metrics (JSON/CSV)
└── thesis/
    └── FedMed-BN_thesis.docx   # Updated thesis
```

---

## ❓ WHAT I NEED FROM YOU NEXT

Once you download the datasets, share with me:
1. **The file format** (CSV? JSON? CoNLL?)
2. **Column names / structure**
3. **A few sample rows** (screenshot or paste)

Then I will write you the **exact, copy-paste-ready code** for each notebook.
