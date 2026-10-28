import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# STEP 1: Define image transformations
transform = transforms.ToTensor()
# transform from the image form to Pytorch tensors from scale 0-255 to 0-1

# STEP 2: Download the MNIST dataset
full_train_dataset = datasets.MNIST(
    root="data",
    # stores the dataset in the data folder
    train=True,
    # select MNIST's training portion
    download=True,
    # download it if it is not availabel
    transform=transform
    # converts each image to a tensor when accessed
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    # access MNIST's separate test portion
    download=True,
    transform=transform
)

# STEP 3: Split training data into training and validation
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [50000, 10000],
    # 50 000 for training and 10 000 for validation
    generator=torch.Generator().manual_seed(42)
)

# STEP 4: Create DataLoaders
# In this project we use the mini-batch SGD, separate the data into small 64 objects each batch
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
    # this is to avoid the dependency in the original ordering
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

def main():
    # STEP 5: Display dataset information
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Testing samples:", len(test_dataset))

    # STEP 6: Inspect a single image
    image, label = full_train_dataset[0]

    print("\nSingle image shape:", image.shape)
    print("Image label:", label)
    print("Minimum pixel value:", image.min().item())
    print("Maximum pixel value:", image.max().item())

    # STEP 7: Inspect a batch
    images, labels = next(iter(train_loader))
    # retrieves the first mini-batch of 64 images and their corresponding labels from train_loader

    print("\nBatch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)

    # STEP 8: Visualize 10 handwritten digits
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        sample_image, sample_label = full_train_dataset[i]

        ax.imshow(sample_image.squeeze(), cmap="gray")
        ax.set_title(f"Label: {sample_label}")
        ax.axis("off")

    plt.suptitle("MNIST Handwritten Digits")
    plt.tight_layout()

    plt.savefig("figures/mnist_samples.png")
    plt.show()


if __name__ == "__main__":
    main()
