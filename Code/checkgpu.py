import torch


def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print("-" * 30)

    is_available = torch.cuda.is_available()
    print(f"CUDA available: {is_available}")

    if is_available:
        try:
            device_count = torch.cuda.device_count()
            print(f"Number of GPUs found: {device_count}")

            current_device = torch.cuda.current_device()
            print(f"Current CUDA device index: {current_device}")

            device_name = torch.cuda.get_device_name(current_device)
            print(f"Current device name: {device_name}")

            print("-" * 30)
            print("GPU IS CORRECTLY DETECTED! Your project is ready to run on the GPU.")
        except Exception as e:
            print(f"\nAn error occurred while getting GPU details: {e}")
            print("This might indicate an issue with the driver or toolkit installation.")
    else:
        print("\nPyTorch cannot detect your GPU. This is likely an installation or driver mismatch issue.")


if __name__ == "__main__":
    check_gpu()

