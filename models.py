import torch
from torch_geometric.nn import GCNConv, MessagePassing,Sequential, NNConv
from torch_geometric.nn import NNConv
import torch.nn as nn


# Ranking module using GNN
# Combining model_diffuse, model_long, model_hidden, model_score to take account of prior interaction,
#  current diffusion, hidden state of nodes, and output probability of a node being infected or not
class Ranking_Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_diffuse= Local_Diffusion()
        self.model_long= Long_Information()
        self.model_hidden = Hidden_State()
        self.model_score = Score_Net()

    def forward(self,data,data_multi,hidden_states):       
        epi_node_features = self.model_diffuse(data)
        info_features= self.model_long(data_multi)
        prev_hidden = hidden_states
        hidden_states = self.model_hidden(epi_node_features,info_features,hidden_states,node_features=data.x)
        scores = self.model_score(prev_hidden,hidden_states,data.x)
        return scores, hidden_states



# Local diffusion for time t, implemented with graph convolution net
class Local_Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        result = self.conv1(x, edge_index,edge_weight = data.edge_attr)
        # result = F.relu(result)
        return result


# Long-range information for multi-graph containing edges of current t and all other t within set window tau
class Long_Information(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nn1 = nn.Sequential(nn.Linear(2, 512),nn.ReLU())
        self.nn2 = nn.Sequential(nn.Linear(2, 4096),nn.ReLU())
        self.l1 = NNConv(8,64, nn = self.nn1,aggr="add")
        self.l2 = NNConv(64,64,nn = self.nn2,aggr="add")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.long()
        x = x.float()
        edge_attr = data.edge_attr.float()
        x = self.l1(x, edge_index,edge_attr = edge_attr)
        x = self.l2(x, edge_index,edge_attr = edge_attr)
        return x


# Update hidden states of nodes based on local diffusion feature and long information feature
class Hidden_State(torch.nn.Module):
    def __init__(self,node=True,GRU=False):
        super().__init__()
        if GRU:
            self.rnn = nn.GRU(64*3+8, 64, 1)
        if node:
            self.nn1 = nn.Sequential(nn.Linear(64*3+8, 64),nn.ReLU())
        else:
            self.nn1= nn.Sequential(nn.Linear(64*3, 64),nn.ReLU())

    def forward(self, epi, info, hidden,node_features = None):
        if node_features is not None:
            x = self.nn1(torch.cat((epi,info,hidden,node_features),axis=1))
        else:
            x = self.nn1(torch.cat((epi,info,hidden),axis=1))
        return torch.nn.functional.normalize(x)


# MLP to output score from hidden states and node features 
class Score_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nn1 = nn.Sequential(nn.Linear(64*2+8, 1))
        self.relu = nn.ReLU()

    def forward(self, prev_hidden, hidden, node_features):
        x = self.nn1(torch.cat((prev_hidden,hidden,node_features),axis=1))
        return self.relu(x)





