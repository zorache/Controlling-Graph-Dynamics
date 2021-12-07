import pandas as pd
import numpy as np
import networkx as nx
from numpy.random import default_rng
import torch
from torch_geometric.utils import barabasi_albert_graph

### Node features ###
# Input
    # either file name of a list of edges 
# Output
    # Each node has static features of betweeness centrality, closeness centrality, eigenvector centrality, degree centrality
    # and dynamic features that is 1 hot for untested, tested positive, tested negative, tested positive in the past 
def get_Node_features(file_name,edges= None):
    if edges is not None:
        G = nx.from_edgelist(edges)
        betweenness= nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        degree = nx.degree_centrality(G)
        eigenvector = nx.eigenvector_centrality(G)

        keys=[int(n) for n in list(betweenness.keys())]
        sorted_keys = sorted(keys)
        betweenness_vec=[]
        closeness_vec=[]
        eigenvector_vec=[]
        degree_vec=[]
        for n in sorted_keys:
            betweenness_vec.append(betweenness[n])
            closeness_vec.append(closeness[n])
            eigenvector_vec.append(eigenvector[n])
            degree_vec.append(degree[n])
    else:
        G = nx.read_edgelist(file_name)
        betweenness= nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        degree = nx.degree_centrality(G)
        eigenvector = nx.eigenvector_centrality(G)

        keys=[int(n) for n in list(betweenness.keys())]
        sorted_keys = sorted(keys)
        betweenness_vec=[]
        closeness_vec=[]
        eigenvector_vec=[]
        degree_vec=[]
        for n in sorted_keys:
            betweenness_vec.append(betweenness[str(n)])
            closeness_vec.append(closeness[str(n)])
            eigenvector_vec.append(eigenvector[str(n)])
            degree_vec.append(degree[str(n)])

    static = np.stack((betweenness_vec,closeness_vec,eigenvector_vec,degree_vec),axis=1)
    dynamic = np.zeros((len(degree),4))
    node_features = np.concatenate((static,dynamic),axis=1)
    return node_features


### Edge features ###
# Input
    # either file name of the setting for preferential attachment graph
# Output
    # Edge index for the desired graph, if the graph's node indices are not continuous, they are mapped to be continuous
def get_Edges(file_name,PA=False,PA_num = None):
    if PA:
        edge_index = barabasi_albert_graph(1000, 5).t().numpy()
    else:
        df = pd.read_csv(file_name, header=None,sep="\t",skiprows=4)
        edges =df.values
        all_nodes = np.unique(edges.T[0])   # We just need to find all unique occurences in one column only since the matrix contains undirected entries (each edge appears twice in reversed direction)
        edge_index = np.zeros(edges.shape)
        dict_map = {k: v for v, k in enumerate(all_nodes)}
        # map to continuous node indices
        for i in range(edge_index.shape[0]):
            for j in range(edge_index.shape[1]):
                edge_index[i,j] = dict_map[edges[i,j]]
    return edge_index
