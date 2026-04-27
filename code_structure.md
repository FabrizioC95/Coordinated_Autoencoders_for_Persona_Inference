# Code Structure

```
src/
├── __init__.py
├── pipeline.py                  # train_model() — orchestrates the full training pipeline
├── model/
│   ├── __init__.py
│   ├── autoencoder.py           # AutoEncoder, KAutoEncoders
│   ├── clustering_head.py       # MixtureAssignmentNetwork
│   └── network.py               # ClusteringAutoEncoder (wrapper)
├── data/
│   ├── __init__.py
│   └── dataloader.py            # NormalDataloader, load_data()
├── training/
│   ├── __init__.py
│   ├── pretrain.py              # shallow_pt_first(), pretrain_mixture_assignment_network()
│   └── trainer.py               # samplewise_trainer()
└── utils/
    ├── __init__.py
    ├── seed.py                  # reset_seed()
    └── inference.py             # run_inference()
```

## Module Descriptions

### `pipeline.py`
Top-level entry point. `train_model()` wires together data loading, pre-training, model initialization, training, and inference into a single callable.

### `model/`
| File | Contents |
|------|----------|
| `autoencoder.py` | `AutoEncoder` — single encoder/decoder pair with configurable depth, batch norm, and dropout. `KAutoEncoders` — stack of k independent autoencoders. |
| `clustering_head.py` | `MixtureAssignmentNetwork` — feed-forward network that outputs soft cluster assignments via Softmax. |
| `network.py` | `ClusteringAutoEncoder` — combines `KAutoEncoders` and `MixtureAssignmentNetwork` into one forward pass. |

### `data/`
| File | Contents |
|------|----------|
| `dataloader.py` | `NormalDataloader` — PyTorch Dataset wrapper. `load_data()` — handles one-hot encoding of categoricals, MinMax scaling of numericals, and returns a configured DataLoader. |

### `training/`
| File | Contents |
|------|----------|
| `pretrain.py` | `shallow_pt_first()` — generates pseudo-labels via KMeans. `pretrain_mixture_assignment_network()` — trains the clustering head as a classification task on pseudo-labels. |
| `trainer.py` | `samplewise_trainer()` — main training loop with weighted reconstruction loss, sample-wise entropy, and batch-wise entropy. Supports `batch` and `epoch` scheduling for the loss coefficients. |

### `utils/`
| File | Contents |
|------|----------|
| `seed.py` | `reset_seed()` — sets seeds for PyTorch, NumPy, Python random, and CUDA. Returns a seeded `torch.Generator` for DataLoader reproducibility. |
| `inference.py` | `run_inference()` — runs the trained model over the full dataset and returns a DataFrame of index-to-cluster assignments. |

## Usage

```python
from src.pipeline import train_model

results_df = train_model(
    data=df,
    k=3,
    categorical_cols=['category1', 'category2'],
    numerical_cols=['num1', 'num2'],
    batch_size=100
)
```
