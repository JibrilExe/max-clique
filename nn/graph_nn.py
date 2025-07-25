"""Graph neural network for the clique problem, the body of the model"""

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    """
    Poging tot GAT gebruik maar op kleine
    grafen leidde deze altijd tot oversmoothing
    """
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 1, heads=1, concat=False)
        self.conv2 = GATConv(1, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        #x = torch.relu(x)
        #x = self.conv2(x, edge_index)
        return x


class GCNConv(MessagePassing):
    """
    Convolutie laag voor graaf netwerk

    Implementatie van torch geometric docs, beetje bijgewerkt zodat geen self loops meer toevoegt,..
    """
    def __init__(self, in_channels, out_channels, x, edge_index):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        self.edge_index = edge_index

        self.row, self.col = self.edge_index
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x):
        x = self.lin(x)

        deg = degree(self.col, x.size(0), dtype=x.dtype) #Normalizatie van data
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[self.row] * deg_inv_sqrt[self.col]

        out = self.propagate(self.edge_index, x=x, norm=norm)
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN(torch.nn.Module):
    """
    Het effectieve GNN model
    """
    def __init__(self, in_channels, out_channels, x, edge_index):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, x, edge_index)
        #self.conv2 = GCNConv(4, out_channels, x, edge_index)
        #self.conv3 = GCNConv(2, in_channels, x, edge_index)
        #self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x) #Uiteindelijk gekozen om met maar 1 laag te werken
        #x = torch.relu(x)
        #x = self.dropout(x)
        #x = self.conv2(x)
        #x = torch.relu(x)
        #x = self.conv3(x)
        return x
