from __future__ import annotations
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training(history: Dict[str, List[float]]):
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(1, max(len(acc), len(val_acc), len(loss), len(val_loss)) + 1)
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    if val_acc:
        plt.plot(epochs, val_acc, "-o", label="val_acc")
    if acc:
        plt.plot(epochs, acc, "-o", label="acc")
    plt.legend(); plt.title("Accuracy")

    plt.subplot(2, 1, 2)
    if val_loss:
        plt.plot(epochs, val_loss, "-o", label="val_loss")
    if loss:
        plt.plot(epochs, loss, "-o", label="loss")
    plt.legend(); plt.title("Loss")
    plt.tight_layout()
