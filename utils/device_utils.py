import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    elif torch.cuda.is_available():
        print("GPU is available")
        print(torch.cuda.get_device_name(0))
    else:
        print("MPS/GPU device not found. Training will continue on CPU.")
        return torch.device("cpu")
