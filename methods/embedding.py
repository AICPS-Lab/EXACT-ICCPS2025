import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, time_steps: int, d_model: int):
        super(TemporalPositionalEmbedding, self).__init__()
        
        # Temporal Embeddings (Trainable)
        self.temporal_embeddings = nn.Parameter(torch.randn(1, time_steps, d_model) * 0.01)
        self.register_parameter('t', self.temporal_embeddings)
        
        # Positional Embeddings (Not trainable)
        self.positional_embeddings = torch.zeros(1, time_steps, d_model)
        position = torch.arange(0., time_steps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        self.positional_embeddings[..., 0::2] = torch.sin(position * div_term)
        self.positional_embeddings[..., 1::2] = torch.cos(position * div_term)
        self.positional_embeddings = nn.Parameter(self.positional_embeddings, requires_grad=False)
        self.register_buffer('pe', self.positional_embeddings)

    def forward(self, x: torch.Tensor):
        """
        x : input data of shape [Batch, T, 6]
        """
        
        # Adding temporal embeddings
        if x.shape[1] != self.temporal_embeddings.shape[1]:
            # x shape is batch x 6x t, need to permute:
            x = x.permute(0, 2, 1)
            x = x + self.temporal_embeddings
            
            # Adding positional embeddings
            x = x + self.positional_embeddings
            # permute back to batch x t x 6
            x = x.permute(0, 2, 1)
        else:
            x = x + self.temporal_embeddings
            x = x + self.positional_embeddings
        return x
    
# Testing the TemporalPositionalEmbedding
if __name__ == "__main__":
    # Data shape [Batch, T, 6]
    dummy_data = torch.randn(32, 100, 6)
    
    model = TemporalPositionalEmbedding(time_steps=100, d_model=6)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, model.state_dict()[name].size())
    state = {'temporal_embeddings': torch.randn(1, 100, 6)}
    out = torch.func.functional_call(TemporalPositionalEmbedding(100, 6), state, (dummy_data))
    # out = model(dummy_data)
    print("Output Shape:", out.shape)  # Should be [32, 100, 6]



