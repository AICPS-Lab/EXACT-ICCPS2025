import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesLinear_Normalq(nn.Module):
    def __init__(self, in_features, out_features, prior_mu=0, prior_sigma=0.1):
        super(BayesLinear_Normalq, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))

    def forward(self, x):
        weight = self.weight_mu + torch.exp(self.weight_sigma) * torch.randn_like(self.weight_sigma)
        bias = self.bias_mu + torch.exp(self.bias_sigma) * torch.randn_like(self.bias_sigma)
        
        kl_divergence = self.kl_divergence()
        
        return F.linear(x, weight, bias), kl_divergence


    def kl_divergence(self):
        weight_var = torch.exp(2 * self.weight_sigma)
        bias_var = torch.exp(2 * self.bias_sigma)
        
        kl_weight = -0.5 * torch.sum(1 + torch.log(weight_var) - self.weight_mu.pow(2) - weight_var)
        kl_bias = -0.5 * torch.sum(1 + torch.log(bias_var) - self.bias_mu.pow(2) - bias_var)
        
        return kl_weight + kl_bias
