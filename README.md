# VisualScribe | Image Captioning with CNN-LSTM
 
A multimodal deep learning system that generates natural language captions for images by combining a **ResNet-50 CNN encoder** for visual feature extraction with an **LSTM decoder** for sequential text generation.
 
> **Portfolio highlight:** Covers computer vision, NLP, model training pipelines, RAG, and quantitative evaluation - all in a single reproducible project.
 
---

![image alt](https://github.com/MostafaAI10/VisualScribe-Image-Captioning-CNN-LSTM/blob/8cc798c134312a4669c8e93f8ed05ad4a7b67c6e/visualscribe_banner.png)
 
## Table of Contents
 
- [Overview](#overview)
- [Features](#features)
- [Figures](#figures)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Next Steps](#next-steps)
- [Author](#author)
 
---
 
## Overview
 
This project bridges computer vision and natural language processing using a classic encoder–decoder architecture:
 
1. **Encoder** : A pretrained ResNet-50 (frozen backbone) extracts a fixed-length feature vector from each input image.
2. **Decoder** : An LSTM network with learned embeddings and dropout generates a caption token-by-token, conditioned on the image features.
3. **Training** : Teacher forcing is used during training with cross-entropy loss and the Adam optimizer.
4. **Evaluation** : Caption quality is measured with BLEU-1 through BLEU-4 scores on a held-out validation set.
 
Compatible datasets include **Flickr8k**, **Flickr30k**, and **MS-COCO**.
 
---
 
## Features
 
| Component | Details |
|-----------|---------|
| Encoder | Pretrained ResNet-50, frozen backbone |
| Decoder | LSTM with embeddings and dropout |
| Vocabulary | NLTK tokenizer, configurable `min_freq` |
| Training | Teacher forcing, Adam optimizer, cross-entropy loss |
| Evaluation | BLEU-1, BLEU-2, BLEU-3, BLEU-4 |
| Outputs | Model checkpoint, vocabulary, training curves, BLEU plots |

---

## Figures
> The figures below are produced automatically in `outputs/` after training.

<img width="1120" height="800" alt="training_curves" src="https://github.com/user-attachments/assets/eea430fc-f899-47a1-946b-88d80bf0fd8f" />

<img width="1120" height="800" alt="bleu_scores" src="https://github.com/user-attachments/assets/4b05f11c-9504-4aa6-9b01-bef952e02824" />

---

## Project Structure
```
image-captioning-cnn-lstm/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ captions.csv        # CSV: image_path, caption, split(train/val/test)
│  └─ images/             # image files
├─ src/
│  ├─ models.py           # EncoderCNN, DecoderLSTM
│  ├─ utils.py            # Vocabulary, dataset, BLEU, collate
│  ├─ train.py            # training loop with checkpoints
│  └─ infer.py            # inference script for generating captions
└─ outputs/
   ├─ best_captioner.pt
   ├─ vocab.json
   ├─ training_curves.png
   ├─ bleu_scores.png
   └─ metrics.json
```
---
 
## Setup
 
**1. Create and activate a virtual environment:**
 
```bash
python -m venv .venv
 
# Windows
.venv\Scripts\activate
 
# Linux / macOS
source .venv/bin/activate
```
 
**2. Install dependencies:**
 
```bash
pip install -r requirements.txt
```
 
**3. Download NLTK tokenizer data (one-time):**
 
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
 
---
 
## Data Preparation
 
1. Place all image files under `data/images/`.
2. Create `data/captions.csv` with the following schema:
 
```csv
image_path,caption,split
images/img1.jpg,A child in a pink dress is climbing up a set of stairs.,train
images/img1.jpg,A little girl climbs into a wooden playhouse.,train
images/img2.jpg,A dog runs through an open grassy field.,val
```
 
The `split` column must contain one of: `train`, `val`, or `test`.
 
Recommended datasets: [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k), [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), [MS-COCO](https://cocodataset.org/).
 
---
 
## Training
 
```bash
python src/train.py \
  --captions data/captions.csv \
  --images-root data \
  --outdir outputs \
  --epochs 10 \
  --batch-size 64 \
  --embed-dim 256 \
  --hidden-dim 512 \
  --min-freq 1 \
  --max-len 20 \
  --lr 1e-3
```
 
### Training Outputs
 
| File | Description |
|------|-------------|
| `outputs/best_captioner.pt` | Best model checkpoint (lowest validation loss) |
| `outputs/vocab.json` | Vocabulary built from the training split |
| `outputs/training_curves.png` | Train/val loss over epochs |
| `outputs/bleu_scores.png` | BLEU-4 score progression |
| `outputs/metrics.json` | Final BLEU scores and loss values |
 
---
 
## Inference
 
Generate a caption for a single image:
 
```bash
python src/infer.py \
  --checkpoint outputs/best_captioner.pt \
  --vocab outputs/vocab.json \
  --image data/images/example1.jpg \
  --max-len 20
```
 
**Example output:**
```
Generated caption: a blue square with the word example on it
```
 
---
 
## Results
 
Training and evaluation visualizations are saved automatically to `outputs/` after each run.
 
- **`training_curves.png`** : Training and validation loss over epochs
- **`bleu_scores.png`** : BLEU-4 score progression across epochs
 
BLEU scores and final loss values are persisted to `metrics.json` for reproducibility.
 
---
 
## Next Steps
 
- [ ] Train for 20+ epochs on a larger dataset (Flickr30k, MS-COCO) for improved caption quality
- [ ] Implement **beam search** decoding to replace greedy inference
- [ ] Unfreeze and fine-tune the CNN backbone for better visual features
- [ ] Add an **attention mechanism** (e.g., Bahdanau or visual attention) to the decoder
- [ ] Evaluate with additional metrics (CIDEr, METEOR, ROUGE-L)
 
---
 
## Author
 MOSTAFA ABDELHAMED | Junior AI & DS Researcher | NVIDIA Gen AI Certified
 [LinkedIn](https://www.linkedin.com/in/mostafa-abdelhamed-88a447286)
