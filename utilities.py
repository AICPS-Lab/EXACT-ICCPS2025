import torch


def get_device():
    """
    Checks for the availability of MPS and CUDA devices and returns the appropriate device.
    If neither is available, returns the CPU device.
    """
    # Check for MPS availability
    if torch.backends.mps.is_available():
        print("MPS is available.")
        return torch.device("mps")

    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available.")
        return torch.device("cuda:1")

    
    # Check the reasons for MPS unavailability
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    elif not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device.")

    print("Neither MPS nor CUDA is available. Using CPU.")
    return torch.device("cpu")