import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32, dropout=0.1, num_mlp_layers=2, GMF=False, MLP=True):
        super(NCF, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.GMF = GMF
        self.MLP = MLP

        # Embedding layers
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim=embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_movies, embedding_dim=embedding_dim)

        # Initialize MLP layers
        MLP_layers = []
        for i in range(self.num_mlp_layers):
            # Input layer shrinks by 2 every time (i times in total). The last layer output has the 
            # desirable embedding dim 
            input_size = self.embedding_dim * (2 ** (self.num_mlp_layers - i))
            MLP_layers.append(nn.Linear(input_size, input_size/2))
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
        self.NeuMF_layer = nn.Sigmoid(NeuMF_size, 1)


        
        self._init_weight_()
        
        # Final prediction
    
    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(movie_ids)

        # GMF layer
        if self.GMF:
            hadamard_product = torch.mul(user_embeds, item_embeds)
            # hadamard_product = self.user_embedding * self.item_embedding
            GMF_output = nn.Linear(self.embedding_dim, 1)(hadamard_product)

        # MLP layer
        if self.MLP:
            MLP_output = self.MLP_layers(torch.cat((self.user_embedding, self.item_embedding), -1)) # -1 as we concat along the last dimension

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
