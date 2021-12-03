import csv
import pandas as pd
import numpy as np
import torch
import networkx as nx


G = nx.read_edgelist("data/ca-GrQc.txt")
betweenness= nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
degree = nx.degree_centrality(G)
eigenvector = nx.eigenvector_centrality(G)
