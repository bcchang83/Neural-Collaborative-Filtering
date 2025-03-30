import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super(NCF, self).__init__()
        # Embedding layers
        
        # GMF
        
        # MLP
        
        # Combine GMF and MLP
        
        # Final prediction
    
    def forward(self, user_ids, movie_ids):
        raise NotImplementedError