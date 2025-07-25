"""This is where we train the models body using the brain"""
import matplotlib.pyplot as plt
import torch
import networkx as nx
from torch_geometric.data import Data
from nn.losses import loss_few_neighbours, loss_neighbors_close, loss_non_neighbors_far, loss_grade
from nn.graph_nn import GCN, GAT

def has_edge(i, j, edge_index):
    """Check if node in edges"""
    return ([i, j] in edge_index or [j, i] in edge_index)

def adjacency(edges, n):
    """
    Berekent de adjacency matrix en complement ervan
    """
    adj = [[0] * n for _ in range(n)]
    compl_adj = [[1] * n for _ in range(n)]
    
    for i, j in edges:
        adj[i][j] = 1
        adj[j][i] = 1
        compl_adj[i][j] = 0
        compl_adj[j][i] = 0

    return torch.tensor(adj, dtype=torch.float), torch.tensor(compl_adj, dtype=torch.float)

def train_test(b, edges, vert, grade):
    """
    Train model print losses and end result
    """

    edge_index = torch.tensor(edges, dtype=torch.long)
    start = []
    for i in range(len(vert)): # We kiezen als startwaarde, de som van de graden van al onze buren
        val = 0
        for edge in edges:
            if edge[0] == i:
                val += grade[edge[1]]
            elif edge[1] == i:
                val += grade[edge[0]]
        start.append([val])
    
    # Normalizeer onze start met het gevonden maximum
    max_val = max([v[0] for v in start])
    vals = [[v[0] / max_val] for v in start]
    
    x = torch.tensor(vals, dtype=torch.float)
    (adj, compl_adj) = adjacency(edges, len(vert))

    # Plotting the graph
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.show()
    """

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    #adj_matrix = to_scipy_sparse_matrix(data.edge_index).todense()

    model = GCN(1, 1, data.x, data.edge_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    model.train()
    losses = []

    epoch = 0
    while (epoch < 100): # (epoch > 1 and (losses[epoch-2] > losses[epoch-1]))):
        optimizer.zero_grad()
        out = model(data.x)
        probabilities = torch.sigmoid(out) # sigmoid voor probabilities tussen 0 en 1

        # Originele termen
        loss1 = torch.mm(probabilities.t(), torch.mm(adj, probabilities))  # p.T * A * p
        loss2 = b*torch.mm(probabilities.t(), torch.mm(compl_adj, probabilities))  # p.T * A_complement * p
        #clique_loss = -torch.sum(probabilities * adj @ probabilities)

        loss = loss2 - loss1
        loss = loss.squeeze() # tensors opgeslaan als [[waarde]], squeeze haalt waarde eruit voor ons
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        #print(f'Epoch: {epoch}, Loss: {loss.item()}')
        epoch += 1

    # Plotting the losses
    """
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()
    """

    # Finaal resultaat
    model.eval()
    out = model(data.x)
    probabilities = torch.sigmoid(out)
    return probabilities
