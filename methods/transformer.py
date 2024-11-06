import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

# from utilities import printc
class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--ninp", type=int, default=64, help="Input channels for the model")
        parser.add_argument("--num_heads", type=int, default=4, help="Number of heads in the transformer")
        parser.add_argument("--embed_dims", type=int, default=256, help="Embedding dimensions in the transformer")
        parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the transformer")
        parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in the transformer")
        parser.add_argument("--init_std", type=float, default=0.02, help="Initialization standard deviation")
        parser.add_argument("--activation", type=str, default='relu', help="Activation function in the transformer")
        return parser
        

    def __init__(self, args):
        num_classes = args.out_channels
        in_channels = args.in_channels
        ninp = args.ninp
        num_heads = args.num_heads
        embed_dims = args.embed_dims
        num_layers = args.num_layers
        dropout = args.dropout
        init_std = args.init_std
        activation = args.activation
        
        super(TransformerModel, self).__init__()
        self.input_emb = nn.Linear(in_channels, ninp)
        self.ninp = ninp
        self.relu = nn.ReLU()
        # modulelist:
        encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=num_heads, dim_feedforward=embed_dims, dropout=dropout, activation=activation, batch_first=True)
        encoder_norm = nn.LayerNorm(ninp)   
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, norm=encoder_norm)
        self.decoder = nn.Linear(ninp, num_classes)
        self.init_std = init_std
        # max layer:
        # self.max = nn.MaxPool1d(100)
        
    def init_weights(self):
        nn.init.trunc_normal_(self.input_emb.weight, std=self.init_std)
        nn.init.trunc_normal_(self.decoder.weight, std=self.init_std)
        # xaiver initialization for Transformer:
        for param in self.transformer_encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
                    
    def forward(self, src):
        src = self.input_emb(src)
        src = self.relu(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
    
    def forward_pred(self, inputs):
        masks = self.forward(inputs)
        masks = masks.permute(0, 2, 1)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred
    
class TransformerClassification(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, num_classes=5, in_channels=6, ninp=64, num_heads=1, embed_dims=256, num_layers=6, dropout=0.1, init_std=.02, activation='relu'):
        super(TransformerClassification, self).__init__()
        self.input_emb = nn.Linear(in_channels, ninp)
        self.ninp = ninp
        self.relu = nn.ReLU()
        # modulelist:
        encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=num_heads, dim_feedforward=embed_dims, dropout=dropout, activation=activation, batch_first=True)
        encoder_norm = nn.LayerNorm(ninp)   
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, norm=encoder_norm)
        self.decoder = nn.Linear(ninp, num_classes)
        self.init_std = init_std
        # max layer:
        # self.max = nn.MaxPool1d(100)
        
    def init_weights(self):
        nn.init.trunc_normal_(self.input_emb.weight, std=self.init_std)
        nn.init.trunc_normal_(self.decoder.weight, std=self.init_std)
        # xaiver initialization for Transformer:
        for param in self.transformer_encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
                    
    def forward(self, src):
        src = self.input_emb(src)
        src = self.relu(src)
        output = self.transformer_encoder(src)
        output = nn.functional.max_pool1d(output.permute(0, 2, 1), kernel_size=output.size(1)).squeeze()
        output = self.decoder(output)
        return output
    
    def forward_pred(self, inputs):
        masks = self.forward(inputs)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred

if __name__ == '__main__':
    # print how many parameters are in the model
    transformer = TransformerClassification(in_channels=6, embed_dims=256)
    print('Number of trainable parameters:', sum(p.numel() for p in transformer.parameters() if p.requires_grad))
    inp = torch.rand(32, 300, 6)
    out = transformer(inp)