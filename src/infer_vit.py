# pylint: disable=C3001,R0914,R0913,R0917,C0115,C0413,C0116,C0301,C0103,E0401,E1101
"""Script for running inference with a trained ViT model."""
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.vit import CustomModel, infer, NUM_CLASSES, DROPOUT, EMBED_DIM, NHEAD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")
MODEL_PATH = "models/vit_latest.pth"


def load_model(ckpt_path: str):
    """Loads a trained CustomModel and its configuration."""
    if not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    dropout = ckpt.get("dropout", DROPOUT)
    embed_dim = ckpt.get("embed_dim", EMBED_DIM)
    nhead = ckpt.get("nhead", NHEAD)
    model_ = CustomModel(
        num_classes=num_classes,
        dropout=dropout,
        embed_dim=embed_dim,
        nhead=nhead,
    ).to(DEVICE)
    model_.load_state_dict(ckpt["model_state"])
    model_.eval()
    print(f"Model loaded from {ckpt_path} and set to evaluation mode.")
    return model_


if __name__ == "__main__":
    image_paths = [
        "Datasets/cifar10_images/test/airplane/test_00003.png",
        "Datasets/cifar10_images/test/airplane/test_00010.png",
        "Datasets/cifar10_images/test/automobile/test_00006.png",
        "Datasets/cifar10_images/test/automobile/test_00009.png",
        "Datasets/cifar10_images/test/bird/test_00025.png",
        "Datasets/cifar10_images/test/bird/test_00035.png",
        "Datasets/cifar10_images/test/cat/test_00000.png",
        "Datasets/cifar10_images/test/cat/test_00008.png",
        "Datasets/cifar10_images/test/deer/test_00022.png",
        "Datasets/cifar10_images/test/deer/test_00026.png",
        "Datasets/cifar10_images/test/dog/test_00012.png",
        "Datasets/cifar10_images/test/dog/test_00016.png",
        "Datasets/cifar10_images/test/frog/test_00004.png",
        "Datasets/cifar10_images/test/frog/test_00005.png",
        "Datasets/cifar10_images/test/horse/test_00013.png",
        "Datasets/cifar10_images/test/horse/test_00017.png",
        "Datasets/cifar10_images/test/ship/test_00001.png",
        "Datasets/cifar10_images/test/ship/test_00002.png",
        "Datasets/cifar10_images/test/truck/test_00014.png",
        "Datasets/cifar10_images/test/truck/test_00011.png",
    ]
    model = load_model(MODEL_PATH)
    correct = 0
    total = 0
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Error: Image not found at {img_path}", file=sys.stderr)
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Failed to load image at {img_path}", file=sys.stderr)
            continue

        pred_idx, pred_class, confidence = infer(model, image)
        true_class = os.path.basename(os.path.dirname(img_path))
        is_correct = pred_class == true_class
        status = "✅" if is_correct else "❌"
        print(
            f"{status} Image: {img_path} | True: {true_class} | Predicted: {pred_class} (class {pred_idx}) | Confidence: {confidence:.4f}"
        )
        total += 1
        if is_correct:
            correct += 1
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")
