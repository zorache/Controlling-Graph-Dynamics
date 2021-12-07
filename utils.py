import numpy as np
from numpy.random import default_rng
import csv

# Create temporal graph
# Input
    # t: how many time steps
    # num_edges: how many edges in the network
    # subset_size: how big the subset to pick from total edges
# Output
    # t_edges where each row is the edges at time t
    # each row of edges at time t consists of [node_1, node_2, prob_transimission]

def sample(t, edges, subset_size):
    num_edges=len(edges)
    t_edges = np.zeros((t,int(subset_size*num_edges),3))
    rng = default_rng()
    for i in range(t):
        vals = rng.integers(low=0, high=len(edges)-1, size=int(num_edges*subset_size))
        edges_sub = edges[vals]
        vals_p = rng.uniform(low = 0.5, high = 1.0, size = int(num_edges*subset_size))
        vals_p = vals_p.reshape((-1, 1))
        combined = np.concatenate((edges_sub,vals_p),axis=1)
        t_edges[i]=combined
    return t_edges


# Generate cumulative multi-graph edges for the information GNN, with all edges including and before the current time step for a given time window
# Input
    # t_edges_index: an array of shape (t, edge_num, 2) which stores the edges for each time step
    # t_edges_attr: an array of shape (t, edge_num) which stores the edge attribute for the edges for each time step
    # t: current time step
    # tau: window to consider prior time steps
# Output
    # multigraph edges
    # multigraph features (the first column is time step difference of the edge compared to current time t)
    #                      (the second column is the transimission attribute of the edge)
def gen_multi_G(t_edges_index,t_edges_attr,t,tau):
    multi_edges = t_edges_index[0:t+1]
    multi_prob = t_edges_attr[0:t+1]
    multi_delay = np.zeros((t+1,len(t_edges_index[0]),))
    for i in range(max(0,t-tau),t+1):
        multi_delay[i]= np.zeros(len(t_edges_index[0]))+(t-i)
    multi_features = np.stack((multi_delay,multi_prob),axis=2)
    return multi_edges.reshape(-1,2), multi_features.reshape(-1,2)

# Generate cumulative multi-graph edges for the information GNN
# def gen_multi_G(t_edges_index,t_edges_attr,t,tau):
#     multi_edges = t_edges_index[0:t+1]
#     multi_prob = t_edges_attr[0:t+1]
#     multi_delay = np.zeros((t+1,len(t_edges_index[0]),))
#     for i in range(max(0,t-tau),t+1):
#         multi_delay[i]= np.zeros(len(t_edges_index[0]))+(t-i)
#     multi_features = np.stack((multi_delay,multi_prob),axis=2)
#     print(multi_features.shape)
#     return multi_edges,multi_features

# Setup initial nodes status based on number of nodes and initial percentage of infection
def status_setup(num_nodes, percent_infect):
    node_status = np.zeros((num_nodes, 4))
    # node_status[:,0]=1
    initial_infect=np.random.binomial(1,percent_infect,size =num_nodes)
    node_status[:,2] = initial_infect
    node_status[:,0]= 1-initial_infect
    return node_status

# Update healthy, latent, infectious, isolated status based on the SEIR framework outlined in the paper
def SEIR_update(node_status,edges, edge_attr,t,delay):
    for i in range(len(edges)):
        u, v = edges [i]
        # Either node is removed/isolated, continue
        if node_status[u][-1]==1 or node_status[v][-1]==1:
            continue 
        # u is infected, v is not 
        # just one direction since graph is undirected
        elif node_status[u][-2]==1 and node_status[v][0]!=0:
            node_status[v][0]= node_status[v][0]*(1-edge_attr[i])
    # Latency period update
    latent_current=node_status[:,1]
    for i in range(len(latent_current)):
        if latent_current[i]>=t:
            node_status[i,1]=-1
            node_status[i,-2]=1
    # Update healthy nodes
    risk = node_status[:,0]
    for j in range(len(risk)):
        if risk[j]!=1 and risk[j]!=0:
            risk[j]=np.random.binomial(1, risk[j])
            if risk[j]==0:
                # Node becomes latent
                node_status[j,1] =  t + np.random.randint(1, delay) 
    node_status[:,0]= risk
    return node_status

# Summarization of current node status
# Input
    # array representing ground truth of nodes, each column is indicting healthy, latent, infectious, isolated, respectively
# Output
    # num of healthy,latent,infected,isolated nodes are returned
def status_summary(node_status):
    total = len(node_status)
    healthy = sum(node_status[:,0]==1)
    latent = sum(node_status[:,1]>0)
    infected = sum(node_status[:,2]==1)
    isolated = sum(node_status[:,-1]==1)
    # print(str(healthy/total)+" percent healthy nodes")
    # print(str(latent/total)+" percent latent nodes")
    # print(str(infected/total)+" percent infected nodes")
    # print(str(isolated/total)+" percent isolated nodes")
    # print(str(healthy)+" healthy nodes")
    # print(str(latent)+" latent nodes")
    # print(str(infected)+" infected nodes")
    # print(str(isolated)+" isolated nodes")
    return healthy,latent,infected,isolated


# Probability for sampling
# Input
    # scores
# Output
    # scores normalized by scores-lowest, divided by sum
def k_node_sample_prob(scores):
    eps = 0.01
    min_score = min(scores)
    scores = scores-min_score+eps
    sum_score = sum(scores)
    scores = scores/sum_score
    return scores



        

# Node for doubly linked list 
class Node:
    def __init__(self, data, prev=None, next=None):
        self.data = data 
        #self.marked = False
        self.prev = prev 
        self.next = next

# Doubly linked list for building adjacency list 
class Doubly_Linked_List:
    def __init__(self):
        self.head = None
        self.len = 0
    def append(self, data):
        node = Node(data)
        if self.head ==None:
            self.head = node
            # self.end = node
            return
        else:
            if self.head.data==data:
                return "Error: adding self loop"
            current = self.head
            while current.next:
                if current.next.data==data:
                    # edge already in adjacency list
                    return
                current = current.next
            node.prev = current
            current.next = node
        self.len +=1
    def delete(self, data):
        deleted = False
        if self.head.data==data:
            return "Error: deleting head of adjacency list"
        else:
            current = self.head.next
            while current:
                if data ==current.data:
                    prev = current.prev
                    next = current.next
                    prev.next = next
                    if next!=None:
                        next.prev = prev
                    deleted=True
                current = current.next
            if deleted:
                self.len -=1
    def print(self):
        current = self.head
        lst = []
        while current:
            lst.append(current.data)
            current = current.next
        print(lst)


# Adjacency list for edge representation, used in heuristic method
class Adjacency_List:
    def __init__(self, file_name,arr=None):
        skip = 4
        self.dict = {}
        if arr is not None:
            for i in range(len(arr)):
                edge = arr[i]
                if edge[0] not in self.dict:
                    lst = Doubly_Linked_List()
                    lst.append(edge[0])
                    self.dict[edge[0]] = lst
                if edge[1] not in self.dict:
                    lst = Doubly_Linked_List()
                    lst.append(edge[1])
                    self.dict[edge[1]] = lst
                self.dict[edge[0]].append(edge[1])
                self.dict[edge[1]].append(edge[0])

        else:
            with open(file_name, newline='') as file:
                if ".txt" in file_name:
                    lines = file.readlines()
                    for line in lines:
                        if skip>0:
                            skip -=1
                            continue
                        edge = (line.strip('\n')).strip('\r').split("\t")
                        if edge[0] not in self.dict:
                            lst = Doubly_Linked_List()
                            lst.append(edge[0])
                            self.dict[edge[0]] = lst
                        if edge[1] not in self.dict:
                            lst = Doubly_Linked_List()
                            lst.append(edge[1])
                            self.dict[edge[1]] = lst
                        self.dict[edge[0]].append(edge[1])
                        self.dict[edge[1]].append(edge[0])
                        
                else:
                    if "musae_git" in file_name:    # graph in prior assignment, not used in this project
                        delimiter = ','
                    else:
                        delimiter=' '
                    reader = csv.reader(file, delimiter=delimiter)
                    for edge in reader:
                        if edge[0] not in self.dict:
                            lst = Doubly_Linked_List()
                            lst.append(edge[0])
                            self.dict[edge[0]] = lst
                        if edge[1] not in self.dict:
                            lst = Doubly_Linked_List()
                            lst.append(edge[1])
                            self.dict[edge[1]] = lst
                        self.dict[edge[0]].append(edge[1])
                        self.dict[edge[1]].append(edge[0])
    
    def delete(self, u):
        if u not in self.dict.keys():
            return "Error: linked list with head "+ u+" does not exist"
        else:
            self.dict.pop(u)
            for n in self.dict.keys():
                self.dict[n].delete(u)

    # Return descending sort of the keys based on length of Adj list
    def sort(self):            
        order={}
        for v in self.dict.values():
            order[v.head.data] = v.len
        return sorted(order, key=lambda k: order[k],reverse=True)
    def print(self):
        for v in self.dict.values():
            v.print()

       
