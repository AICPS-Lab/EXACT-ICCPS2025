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

class sliding_windows(torch.nn.Module):
    def __init__(self, width, step):
        # https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
        super(sliding_windows, self).__init__()
        self.width = width
        self.step = step

    def forward(self, input_time_series, labels):
        input_transformed = torch.swapaxes(input_time_series.unfold(-2, size=self.width, step=self.step), -2, -1)
        # For labels, we only have one dimension, so we unfold along that dimension
        labels_transformed = labels.unfold(0, self.width, self.step)
        return input_transformed, labels_transformed

    def get_num_sliding_windows(self, total_length):
        return round((total_length - (self.width - self.step)) / self.step)