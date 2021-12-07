from utils import *
from preprocessing import *
import numpy as np
from heuristics import *
import os


heuristic = "hop"              #OR "half_incubation", "hop"
dataset = "data/ca-GrQc.txt"   #OR "PA_5", "PA_3", "data/ca-GrQc.txt" 

T = 15              # num of time steps 
subset_size = 0.3   # at each time step, subset_size of the edges interacted from the whole graph 
infect_rate = 0.05  # initial infection rate of total population 
k = 100             # num of interventions
delay =5            # incubation period is determined by random variable for int: 1 to delay-1

if dataset=="data/ca-GrQc.txt":
    edges = get_Edges("data/ca-GrQc.txt")
    if os.path.exists('node_features.npy'):
        with open('node_features.npy', 'rb') as f:             
            node_features = np.load(f) 
    else:
        edges = get_Edges("data/ca-GrQc.txt")
elif "PA" in dataset:
    edges = get_Edges("",PA=True,PA_num = int(dataset[-1]))
    node_features = get_Node_features("",edges=edges)


node_status = status_setup(len(node_features),infect_rate)
temporal_edges = sample(T, edges, subset_size)
t_edges_index = temporal_edges[:,:,0:2].astype(int)
t_edges_attr =  temporal_edges[:,:,2]


infected_index = np.where(node_status[:,2] == 1)[0]

detection_rate = np.zeros(T)
total_summary_healthy = np.zeros(T)

if heuristic=="half_incubation":
    initial = np.random.randint(2,size = len(node_status))

for t in range(T):
    print("-------------------")
    print("t = ",t)
    if t!=0:
        SEIR_update(node_status,t_edges_index[t-1], t_edges_attr[t-1],t,delay)
    status_summary(node_status)
    if heuristic=="half_incubation":
        test_lst = half_incubation(initial,t,k)
    else:
        adj_lst = Adjacency_List("",arr = t_edges_index[t])
        test_lst = infected_neigh(adj_lst,infected_index,k)
    for n in test_lst:
        if node_status[n,2]==1:
            node_status[n,2]=0
            node_status[n,-1]=1
            detection_rate[t]+=1
        elif node_status[n,1]>0:
            node_status[n,1]=0
            node_status[n,-1]=1
            detection_rate[t]+=1
    total_summary_healthy[t],_,_,_ = status_summary(node_status)

detection_rate = detection_rate/k
print(total_summary_healthy/len(node_status))
print(detection_rate)


