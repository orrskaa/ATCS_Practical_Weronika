# ATCS Practical – Sentence Embeddings with SNLI and SentEval

This repository contains the implementation of multiple neural models for Natural Language Inference (NLI) using the SNLI dataset. It also evaluates sentence embeddings using the SentEval toolkit across various transfer tasks. The SNLI dataset consists of 570,000 labeled sentence pairs with three balanced classes: entailment, contradiction, and neutral. Each model uses fixed 300-dimensional GloVe word embeddings pretrained on Common Crawl (840B tokens).

---

## Requirements & Setup

### Clone and Environment Setup

```bash
git clone https://github.com/orrskaa/ATCS_Practical_Weronika.git
cd ATCS_practical
python3 -m venv atcs_env
source atcs_env/bin/activate
pip install -r requirements.txt
```

### Additional Dependencies

- Install `torch`, `transformers`, `nltk`, `senteval`, and `tensorboard`
- Make sure to run once:

```python
import nltk
nltk.download('punkt')
```

### Download GloVe Embeddings

Upload the GloVe 840B 300d vectors in `data/glove.840B.300d.txt`. You can download them from:
https://nlp.stanford.edu/projects/glove/

### Download SentEval Tasks

```bash
cd ATCS_practical
mkdir -p senteval_data
cd senteval_data
git clone https://github.com/facebookresearch/SentEval.git .
cd data/downstream
./get_transfer_data.bash
```

Update `senteval_eval.py` to point to this directory:
```python
PATH_TO_DATA = os.path.join(os.getcwd(), 'senteval_data/data')
```

---

## Code Structure

```
ATCS_practical/
├── models/                     # All model definitions
│   ├── avg_glove.py
│   ├── lstm_uni_classifier.py
│   ├── lstm_bi_classifier.py
│   └── lstm_bi_maxpool_classifier.py
├── utils/                      # Utility modules
│   ├── glove_loader.py
│   ├── vocab_builder.py
│   ├── collate.py
│   └── preprocessing.py
├── scripts/                    # (Optional) for extended scripts
├── logs/                       # SLURM log outputs and errors
├── checkpoints/                # Saved model checkpoints
├── tests/                      # Pytest-based unit tests
├── notebooks/                  # Any optional notebooks
├── train.py                    # Training pipeline
├── eval.py                     # Evaluation on SNLI test set
├── senteval_eval.py            # Evaluation on SentEval tasks
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```

---

## Models Implemented

| Model         | Description                       |
|---------------|-----------------------------------|
| `avg_glove`   | Averages word embeddings (GloVe)  |
| `lstm`        | Uni-directional LSTM encoder      |
| `bilstm`      | Bi-directional LSTM with concat   |
| `bilstm_max`  | BiLSTM with max-pooling over time |

---

## Train Models

```bash
python train.py --model <model_name>
--> example: python train.py --model avg_glove
# Or use SLURM:
sbatch run_avg_glove.sh
```

Model checkpoints are saved to `checkpoints/<model_name>.pt`.

---

## Evaluate on SNLI Test Set

```bash
python eval.py --model <model_name> --checkpoint checkpoints/<model_name>.pt
--> example: python eval.py --model avg_glove --checkpoint checkpoints/avg_glove.pt
# Or via SLURM:
sbatch eval_avg_glove.sh
```

---

## Evaluate with SentEval

```bash
python senteval_eval.py --model <model_name> --checkpoint checkpoints/<model_name>.pt
--> example: python eval.py --model avg_glove --checkpoint checkpoints/avg_glove.pt
# Or via SLURM:
sbatch eval_senteval_avg_glove.sh
```

Results are saved to `senteval_results_<model_name>.txt`.

---

## TensorBoard Visualizations

Training logs are saved to `runs/`.

```bash
tensorboard --logdir=runs
```

 [TensorBoard Screenshots (Google Drive)](https://drive.google.com/drive/folders/1Uyn8ah_Q7cqe3bnAHNgaO1Ev7XHtfFYU)

---

## Deliverables

- Implemented 4 models for NLI using GloVe + LSTM variations
- Evaluated models on SNLI test set and SentEval tasks
- TensorBoard logging
- Modular and extensible codebase
- Unit tests for preprocessing & utilities

---

## Author
Weronika Orska
Univerity of Amsterdam
