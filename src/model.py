import torch
import torch.nn as nn

class DigitRecognitionNN(nn.Module): # Every PyTorch network typically inherits from this nn.Module

    def __init__(self):
        super().__init__()

        # Flatten 28x28 image into 784 values for the initial layer because it had 784 neurons already
        self.flatten = nn.Flatten() # [64, 1, 28, 28] --> [64, 784]

        # First hidden layer
        self.layer1 = nn.Linear(784, 256)

        # Second hidden layer
        self.layer2 = nn.Linear(256, 128)

        # Output layer
        self.output_layer = nn.Linear(128, 10)

        # Activation function
        self.relu = nn.ReLU() # ReLu allows neural network to learn nonlinear realtionships

    def forward(self, x):

        # Convert image to vector
        x = self.flatten(x)

        # Hidden layer 1
        x = self.layer1(x)
        x = self.relu(x)

        # Hidden layer 2
        x = self.layer2(x)
        x = self.relu(x)

        # Output layer
        x = self.output_layer(x)

        return x


def main():

    # Create the neural network
    model = DigitRecognitionNN()

    print("Neural Network Architecture:")
    print(model)

    # Create a fake mini-batch of 64 images
    sample_input = torch.randn(64, 1, 28, 28)

    # Forward propagation
    output = model(sample_input)

    print("\nInput Shape:", sample_input.shape)
    print("Output Shape:", output.shape)

    # Count trainable parameters
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\nTotal Trainable Parameters:", total_parameters)

    # Show output scores for first image
    print("\nFirst Image Class Scores:")
    print(output[0])

    # Get predicted digit
    prediction = torch.argmax(output[0]).item()

    print("\nPredicted Digit:", prediction)


if __name__ == "__main__":
    main()
