import torch
import torch.nn as nn
import numpy as np

class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, dropout=0.1, num_mlp_layers=2, GMF=True, MLP=True):
        super(NCF, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.GMF = GMF
        self.MLP = MLP

        # Embedding GMF layers
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim=embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_movies, embedding_dim=embedding_dim)

        # Initialize MLP layers
        self.user_embedding_MLP = nn.Embedding(num_users, embedding_dim * (2 ** (num_mlp_layers - 1)))
        self.item_embedding_MLP = nn.Embedding(num_movies, embedding_dim * (2 ** (num_mlp_layers - 1)))
        
        MLP_layers = []
        for i in range(self.num_mlp_layers):
            # Input layer shrinks by 2 every time (i times in total). The last layer output has the 
            # desirable embedding dim 
            input_size = self.embedding_dim * (2 ** (self.num_mlp_layers - i))
            MLP_layers.append(nn.Linear(input_size, input_size//2))
            MLP_layers.append(nn.Dropout(self.dropout))
            MLP_layers.append(nn.ReLU())
            
        self.MLP_layers = nn.Sequential(*MLP_layers)

        # Create NeuMF layer
        if self.GMF and self.MLP:
            NeuMF_size = self.embedding_dim * 2
        elif (self.GMF or self.MLP):
            NeuMF_size = self.embedding_dim 
        else:
            raise('Please use at least one model, either GMF or MLP')
        # Final prediction
        self.NeuMF_layer = nn.Sequential(
                nn.Linear(NeuMF_size, 1),
                nn.Sigmoid()
                )
        


        
        # self._init_weight_()
        
        
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(movie_ids)

        user_embeds_MLP = self.user_embedding_MLP(user_ids)
        item_embeds_MLP = self.item_embedding_MLP(movie_ids)

        # GMF layer
        if self.GMF:
            # hadamard_product = torch.mul(user_embeds, item_embeds)
            hadamard_product = user_embeds * item_embeds
            GMF_output = hadamard_product #nn.Linear(self.embedding_dim, 1)(hadamard_product)

        # MLP layer
        if self.MLP:
            # print(f"user shape={user_embeds_MLP.shape}\nmovie shape={item_embeds_MLP.shape}\n")
            MLP_output = self.MLP_layers(torch.cat((user_embeds_MLP, item_embeds_MLP), -1)) # -1 as we concat along the last dimension
            
        if self.GMF and self.MLP:
            NeuMF_input = torch.cat((GMF_output, MLP_output), -1)
        elif self.GMF:
            NeuMF_input = GMF_output
        elif self.MLP:
            NeuMF_input = MLP_output
        else:
            raise('Please use at least one model, either GMF or MLP')

        NeuMF_out = self.NeuMF_layer(NeuMF_input)

        return NeuMF_out.view(-1)  
