import torch
import torch.nn as nn

from dataset import train_loader
from model import DigitRecognitionNN


def main():

    # STEP 1: Make results reproducible
    torch.manual_seed(42) # with same setup the initial weights will be the same each run but the results are not guaranteed the same

    # STEP 2: Create the neural network
    model = DigitRecognitionNN()

    # STEP 3: Define the loss function
    criterion = nn.CrossEntropyLoss()

    # STEP 4: Retrieve one mini-batch from MNIST
    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    # STEP 5: Forward propagation
    outputs = model(images)
    # passes 64 MNIST images through the neural network [64, 1, 28, 28] --> [64, 10]

    print("\nModel output shape:", outputs.shape)

    # STEP 6: Calculate Cross-Entropy Loss
    loss = criterion(outputs, labels)

    print("\nLoss before training:", loss.item())

    # STEP 7: Clear existing gradients
    model.zero_grad() # we must clear the existing gradients before computing the new one so it won't combine with the previous one

    # STEP 8: Backpropagation
    loss.backward() # calcualte the weights, the gradient is not updated yet

    # STEP 9: Inspect gradients
    print("\nGradient information:")

    for name, parameter in model.named_parameters():
        if parameter.grad is not None: # Pytorch save these gradients in parameter.grad
            print(
                f"{name}: "
                f"shape={parameter.grad.shape}, "
                f"mean_absolute_gradient="
                f"{parameter.grad.abs().mean().item():.6f}"
            )

    # STEP 10: Get predictions
    predictions = torch.argmax(outputs, dim=1)

    # STEP 11: Calculate accuracy
    correct = (predictions == labels).sum().item()

    accuracy = correct / len(labels) * 100

    print("\nCorrect predictions:", correct)
    print("Total images:", len(labels))
    print(f"Mini-batch accuracy: {accuracy:.2f}%")

    # STEP 12: Compare first 10 predictions
    print("\nFirst 10 predictions vs labels:")

    for i in range(10):
        print(
            f"Predicted: {predictions[i].item()} | "
            f"Actual: {labels[i].item()}"
        )


if __name__ == "__main__":
    main()
