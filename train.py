import torch
from torch_geometric.loader import DataLoader,RandomNodeSampler, NeighborLoader
from torch_geometric.data import Data
from models import *
from utils import *
from preprocessing import *
import networkx as nx
import os

epochs = 30
T = 15                       # num of time steps 
dataset="data/ca-GrQc.txt"   # OR "PA_5", "PA_3"
subset_size=0.3              # at each time step, subset_size of the edges interacted from the whole graph 
infect_rate=0.01             # initial infection rate of total population 
k = 100                      # num of interventions
delay =5                     # incubation period is determined by random variable for int: 1 to delay-1
tau = 7                      # window considered for multi-graph input into long information GNN

name_of_run = 'model_ranking_30_initial_infect_given_arg_max_collab.pth'    # name that the model will be saved as 


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model_ranking = Ranking_Module().to(device)
optimizer = torch.optim.Adam(model_ranking.parameters(), lr=0.001, weight_decay=5e-4)

train_running_loss = 0

for i in range(epochs):
    print("-------------------")
    print("Epoch",i)
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
    

    hidden_states = torch.rand(len(node_features), 64)
    hidden_states= hidden_states.to(device)

    loss_fc = nn.BCELoss() 

    node_features = torch.from_numpy(node_features).to(device).float()
    # All nodes are untested from the beginning
    node_features[:,4]  = node_features[:,4] + 1 
    
    # For each epoch, we randomly sample from the PA graph and construct new node statuses
    node_status = status_setup(len(node_features),infect_rate)
    temporal_edges = sample(T, edges, subset_size)
    t_edges_index = temporal_edges[:,:,0:2].astype(int)
    t_edges_attr =  temporal_edges[:,:,2]

    # Given initial infected nodes
    infected_index = np.where(node_status[:,2] == 1)[0]
    node_features[infected_index,4]  = 0  
    node_features[infected_index,5]  = 1 
    model_ranking.train()
    for t in range(T):
        print("-------------------")
        print("t = ",t)
        if t!=0:
            SEIR_update(node_status,t_edges_index[t-1], t_edges_attr[t-1],t,delay)
        #status_summary(node_status)
        edges_index = torch.from_numpy(t_edges_index[t]).to(device).long()
        edges_attr = torch.from_numpy(t_edges_attr[t]).to(device).float()
        data = Data(x = node_features, edge_index=edges_index.t().contiguous(),edge_attr= edges_attr)
        
        multi_edges, multi_attr = gen_multi_G(t_edges_index,t_edges_attr,t,tau)
        multi_edges = torch.from_numpy(multi_edges).to(device)
        multi_attr  = torch.from_numpy(multi_attr).to(device)
        data_multi = Data(x = node_features, edge_index=multi_edges.t().contiguous(),edge_attr= multi_attr)
        
        scores, hidden_states = model_ranking(data,data_multi,hidden_states)
        hidden_states = hidden_states.detach()
        
        labels = 1-node_status[:,0]           
        labels = torch.from_numpy(labels)
        labels = labels.reshape(-1,1)
        labels = labels.to(device).float()


        loss = loss_fc(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        probability = k_node_sample_prob(scores)
        counter =k
        isolated_index = np.where(node_status[:,3] == 1)[0]

        probability[isolated_index]=0
        while counter>0:
            prev_test_positive = torch.where(node_features[:,5]==1)[0]
            node_features[prev_test_positive,5]=0
            node_features[prev_test_positive,7]=1
            #index = np.argmax(np.random.multinomial(1,probability.cpu().detach().numpy().reshape(-1,),1))
            index = torch.argmax(probability)
            node_features[index,4]=0 
            if node_status[index,2]==1:
                node_status[index,2]=0
                node_status[index,-1]=1
                node_features[index,5]=1
            elif node_status[index,1]>0:
                node_status[index,1]=0
                node_status[index,-1]=1
                node_features[index,5]=1
            else:
                node_features[index,6]=1
            probability[index]=0
            counter-=1
        train_running_loss += loss.detach().item()
    if i%10==0 and i!=0:
        # Validate 
        model_ranking.eval()
        if dataset=="data/ca-GrQc.txt":
            edges = get_Edges("data/ca-GrQc.txt")
            with open('node_features.npy', 'rb') as f:
                node_features = np.load(f) 
        elif "PA" in dataset:
            edges = get_Edges("",PA=True,PA_num = int(dataset[-1]))
            node_features = get_Node_features("",edges=edges)


        hidden_states = torch.rand(len(node_features), 64)
        hidden_states= hidden_states.to(device)


        node_features = torch.from_numpy(node_features).to(device).float()
        # All nodes are untested from the beginning
        node_features[:,4]  = node_features[:,4] + 1 

        node_status = status_setup(len(node_features),infect_rate)
        temporal_edges = sample(T, edges, subset_size)
        t_edges_index = temporal_edges[:,:,0:2].astype(int)
        t_edges_attr =  temporal_edges[:,:,2]

        #Given initial infected nodes
        infected_index = np.where(node_status[:,2] == 1)[0]
        node_features[infected_index,4]  = 0  
        node_features[infected_index,5]  = 1 

        total_summary_healthy = np.zeros(T)
        with torch.no_grad():
            for t in range(T):
                print("--------Validating-----------")
                print("t = ",t)
                if t!=0:
                    SEIR_update(node_status,t_edges_index[t-1], t_edges_attr[t-1],t,delay)
                #status_summary(node_status)
                edges_index = torch.from_numpy(t_edges_index[t]).to(device).long()
                edges_attr = torch.from_numpy(t_edges_attr[t]).to(device).float()
                data = Data(x = node_features, edge_index=edges_index.t().contiguous(),edge_attr= edges_attr)
                        
                multi_edges, multi_attr = gen_multi_G(t_edges_index,t_edges_attr,t,tau)
                multi_edges = torch.from_numpy(multi_edges).to(device)
                multi_attr  = torch.from_numpy(multi_attr).to(device)
                data_multi = Data(x = node_features, edge_index=multi_edges.t().contiguous(),edge_attr= multi_attr)
                
                scores, hidden_states = model_ranking(data,data_multi,hidden_states)

                data.detach()
                data_multi.detach()
                hidden_states = hidden_states.detach()
                # Pick top k that are not isolated 
                probability = k_node_sample_prob(scores)
                counter =k
                isolated_index = np.where(node_status[:,3] == 1)[0]

                probability[isolated_index]=0
                while counter>0:
                    prev_test_positive = torch.where(node_features[:,5]==1)[0]
                    node_features[prev_test_positive,5]=0
                    node_features[prev_test_positive,7]=1
                    #index = np.argmax(np.random.multinomial(1,probability.cpu().detach().numpy().reshape(-1,),1))
                    index = torch.argmax(probability)
                    node_features[index,4]=0 
                    if node_status[index,2]==1:
                        node_status[index,2]=0
                        node_status[index,-1]=1
                        node_features[index,5]=1
                    elif node_status[index,1]>0:
                        node_status[index,1]=0
                        node_status[index,-1]=1
                        node_features[index,5]=1
                    elif node_status[index,-1]==1:
                        continue
                    else:
                        node_features[index,6]=1
                    probability[index]=0
                    counter-=1
                total_summary_healthy[t],_,_,_ = status_summary(node_status)
            print(total_summary_healthy/len(node_status))
        
torch.save(model_ranking.state_dict(), name_of_run)
