import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path

from dataset import train_loader, val_loader
from model import DigitRecognitionNN


# STEP 1: Training configuration
torch.manual_seed(42)

LEARNING_RATE = 0.01
EPOCHS = 20 # one complete pass through the entire training dataset, the whole dataset has been through the neural network for 1 epoch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# STEP 2: Initialize model
model = DigitRecognitionNN().to(DEVICE)

# STEP 3: Define loss function
criterion = nn.CrossEntropyLoss()

# STEP 4: Define optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE
)


# STEP 5: Training function
def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss and accuracy
        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy


# STEP 6: Validation function
def validate(): # This one is used to adjust the hyperparameters which are the learning rate, model complexity,..

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = 100 * correct / total

    return average_loss, accuracy


# STEP 7: Main training loop
def main():

    Path("models").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0

    print("Training on:", DEVICE)

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc

            torch.save(
                model.state_dict(),
                "models/best_model.pth"
            )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

    # STEP 8: Plot loss curves
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("figures/loss_curve.png")
    plt.close()

    # STEP 9: Plot accuracy curves
    plt.figure(figsize=(8, 5))

    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig("figures/accuracy_curve.png")
    plt.close()

    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print("Best model saved to models/best_model.pth")


if __name__ == "__main__":
    main()
