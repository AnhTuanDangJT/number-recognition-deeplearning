import torch
import torchvision


def main():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)

    if torch.cuda.is_available():
        print("Device available: CUDA GPU")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Device available: CPU")

    x = torch.tensor([1.0, 2.0, 3.0])

    print("\nTest tensor:")
    print(x)

    print("\nTensor shape:")
    print(x.shape)

    print("\nSetup successful!")


if __name__ == "__main__":
    main()