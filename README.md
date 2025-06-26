# 👀 Vision Models Explored 🔥

This project explores different neural network models (MLP, CNN, ViT) for classifying images from the CIFAR-10 dataset using PyTorch. Learn how to train, evaluate, and run inference with these models.

---

## Project Structure

```
Vision-Models-Explored/
├── Datasets/                # Place CIFAR-10 images here (see below)
├── logging/                 # Training logs and plots
├── models/                  # Saved model checkpoints
├── src/
│   ├── cnn.py               # CNN model and training script
│   ├── mlp.py               # MLP model and training script
│   ├── vit.py               # Vision Transformer model and training script
│   ├── infer_cnn.py         # Inference script for CNN
│   ├── infer_mlp.py         # Inference script for MLP
│   └── infer_vit.py         # Inference script for ViT
├── tools/                   # Utilities and notebooks
├── requirements.txt
└── README.md
```

---

## Setup

1. **Set up the environment:**

   ```sh
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```

2. **Download the CIFAR-10 dataset:**

   - Open `tools/cifar_dl.ipynb` in Jupyter or VSCode.
   - This will download the CIFAR-10 dataset and save the images into `Datasets/cifar10_images/` with the following structure:
     ```
     Datasets/cifar10_images/
       ├── train/
       │   ├── airplane/
       │   ├── automobile/
       │   └── ...
       └── test/
           ├── airplane/
           ├── automobile/
           └── ...
     ```

---

## Training

Run the script for the model you want to train:

- **MLP:**
  ```sh
  python src/mlp.py
  ```

- **CNN:**
  ```sh
  python src/cnn.py
  ```

- **Vision Transformer (ViT):**
  ```sh
  python src/vit.py
  ```

Checkpoints and plots will be saved in `models/` and `logging/`.

---

## Inference

Run inference on test images:

- **MLP:**
  ```sh
  python src/infer_mlp.py
  ```

- **CNN:**
  ```sh
  python src/infer_cnn.py
  ```

- **ViT:**
  ```sh
  python src/infer_vit.py
  ```

---

## Models

- **MLP (`src/mlp.py`):**  
  A simple multilayer perceptron that flattens the image and passes it through fully connected layers.

- **CNN (`src/cnn.py`):**  
  A basic convolutional neural network with two convolutional layers and a fully connected output.

- **Vision Transformer (`src/vit.py`):**  
  A minimal ViT implementation that splits the image into patches, embeds them, adds positional encoding, and processes them with a transformer encoder.

All models use the same training and evaluation pipeline for fair comparison.

---

## Results & Logging

- Training and validation loss/accuracy plots are saved in `logging/`.
- Model checkpoints are saved in `models/` with timestamps and as `*_latest.pth`.
- Example inference results are printed to the console.
