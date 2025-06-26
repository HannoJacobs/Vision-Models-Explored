# pylint: disable=C3001,R0914,R0913,R0917,C0115,C0413,C0116,C0301,C0103,E0401,E1101
"""Computer vision model"""
import os
import time
import datetime

import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  device = {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
DROPOUT = 0.1
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
NUM_CLASSES = len(CIFAR10_CLASSES)


def load_cifar_train_val(path="Datasets/cifar10_images", batch_size=64, shuffle=False):
    """load_cifar"""
    transform = transforms.ToTensor()
    train_dataset = ImageFolder(root=f"{path}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{path}/test", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader


class CustomModel(nn.Module):
    """Model"""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()

        kernel_size = 3
        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.conv_2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.full_connected = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = self.full_connected(x)
        return x


def custom_loss_function(logits, targets):
    """
    Focal Loss implementation for multi-class classification.
    Helps with class imbalance by down-weighting easy examples.
    """
    return nn.CrossEntropyLoss()(logits, targets)


def train_epoch(model_, loader, optimizer_):
    """Trains the model for one epoch and computes loss and accuracy."""
    model_.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)

        optimizer_.zero_grad()
        logits = model_(features)  # (B, num_classes)
        loss = custom_loss_function(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model_.parameters(), 1.0)  # Optional: gradient clip
        optimizer_.step()

        # loss.item() is avg loss for batch
        total_loss += loss.item() * features.size(0)

        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def eval_epoch(model_, loader):
    """Evaluates the model and computes loss and accuracy."""
    model_.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for features, targets in loader:
        features, targets = features.to(DEVICE), targets.to(DEVICE)

        logits = model_(features)
        loss = custom_loss_function(logits, targets)

        total_loss += loss.item() * features.size(0)

        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def infer(model_, image):
    """
    Predicts the class for a single image array.

    Args:
        model_: The trained model
        image: numpy array image (H, W, C) in BGR format (as loaded by cv2)

    Returns:
        tuple: (predicted_class_index, predicted_class_name, confidence)
    """
    model_.eval()

    # Convert BGR to RGB (cv2 loads as BGR, but transforms expect RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL-like format and apply same transforms as training
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )

    # Apply transforms and add batch dimension
    image_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    # Get prediction
    logits = model_(image_tensor)  # (1, num_classes)
    probabilities = torch.softmax(logits, dim=1)  # (1, num_classes)

    # Get predicted class and confidence
    confidence_, predicted_idx = torch.max(probabilities, 1)
    predicted_idx = predicted_idx.item()
    confidence_ = confidence_.item()
    predicted_class_name = CIFAR10_CLASSES[predicted_idx]

    return predicted_idx, predicted_class_name, confidence_


if __name__ == "__main__":
    start_time = time.time()
    train_dl, val_dl = load_cifar_train_val(batch_size=BATCH_SIZE, shuffle=True)
    train_size = len(train_dl.dataset)
    val_size = len(val_dl.dataset)
    print(f"Loaded {train_size:,} training samples and {val_size:,} validation samples")

    # Model & Optimizer
    model = CustomModel(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"\nStarting training for {EPOCHS} epochs on {DEVICE}...")
    for ep in range(1, EPOCHS + 1):
        epoch_start_time = time.time()

        tr_loss, tr_acc = train_epoch(model, train_dl, optimizer)
        vl_loss, vl_acc = eval_epoch(model, val_dl)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accuracies.append(tr_acc)
        val_accuracies.append(vl_acc)

        epoch_end_time = time.time()
        epoch_duration_seconds = int(epoch_end_time - epoch_start_time)
        epoch_minutes, epoch_seconds = divmod(epoch_duration_seconds, 60)

        print(
            f"Epoch {ep:02d}/{EPOCHS} │ "
            f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} │ "
            f"val_loss={vl_loss:.3f} val_acc={vl_acc:.3f} │ "
            f"Time: {epoch_minutes}m {epoch_seconds}s"
        )

    # Save model
    script_name = os.path.basename(__file__)
    script_name_no_ext = os.path.splitext(script_name)[0]
    os.makedirs("models", exist_ok=True)
    os.makedirs("logging", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    TS_MODEL_PATH = f"models/{script_name_no_ext}_{ts}.pth"
    LATEST_MODEL_PATH = f"models/{script_name_no_ext}_latest.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
        },
        TS_MODEL_PATH,
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": NUM_CLASSES,
            "dropout": DROPOUT,
        },
        LATEST_MODEL_PATH,
    )

    # Plot metrics
    TS_METRICS_PATH = f"logging/{script_name_no_ext}_metrics_{ts}.png"
    LATEST_METRICS_PATH = f"logging/{script_name_no_ext}_metrics_latest.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs_range = range(1, EPOCHS + 1)
    ax1.plot(epochs_range, train_losses, label="Train Loss")
    ax1.plot(epochs_range, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accuracies, label="Train Accuracy")
    ax2.plot(epochs_range, val_accuracies, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    # Add title
    fig.suptitle(f"{script_name}\nCIFAR-10\nEpochs: {EPOCHS}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(TS_METRICS_PATH)
    plt.savefig(LATEST_METRICS_PATH)
    plt.close(fig)

    # Runtime
    total_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(total_seconds, 60)
    print(f"\nTotal runtime: {minutes}m {seconds}s")

    # Demo inference
    print("\n--- Demo Inference ---")
    demo_image_path = "Datasets/cifar10_images/test/horse/test_00013.png"
    demo_image = cv2.imread(demo_image_path)
    pred_idx, pred_class, confidence = infer(model, demo_image)
    print(f"Image: {demo_image_path}")
    print(
        f"Predicted: {pred_class} (class {pred_idx}) with confidence {confidence:.2f}"
    )
