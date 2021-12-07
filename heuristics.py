from utils import *


# Heuristic method of ranking nodes to be intervened on based on number of infected neighbors (1-hop and 2-hop)
# input
    # adj_lst: adjacency list
    # infected: list of known infected nodes
    # k: number of nodes picked for intervention
# output
    # list of k nodes 
def infected_neigh(adj_lst,infected,k, feature=False):
    total = np.zeros((len(adj_lst.dict),2))
    for i in range(len(total)):
        if str(i) not in adj_lst.dict:
            continue
        if i not in adj_lst.dict:
           continue
        hop_1 = adj_lst.dict[i].head.next
        while hop_1:
            hop_2=adj_lst.dict[hop_1.data].head.next
            while hop_2:
                if int(hop_2.data) in infected:
                    total[i,1] +=1
                hop_2=hop_2.next
            if int(hop_1.data) in infected:
                total[i,0] +=1
            hop_1=hop_1.next
    if feature:
        return total
    total = [list(n) for n in total]    #turn into list
    order = sorted(range(len(total)),key=total.__getitem__,reverse=True)
    test = []
    for i in range(len(order)):
        if order[i] not in infected:
            test.append(order[i])
            k-=1
            if k==0:
                break
    return test

# Heuristic method of testing each node every 2 time step
# note this is testing half the expected incubation period (same as BU testing)    
# randomization is included by shuffling the nodes such that only random k tested (some students may be behind schedule)
# input
    # initial: indicator list of whether a node is supposed to test at initial step  
    # t: current time step
    # k: number of nodes picked for intervention
# output
    # list of k nodes                 

def half_incubation(initial,t,k):
    if t%2 == 0:
        test_lst = np.where(initial==1)[0]
        np.random.shuffle(test_lst)
        test_lst= test_lst[:k]
    else:
        test_lst = np.where(initial==0)[0]
        np.random.shuffle(test_lst)
        test_lst= test_lst[:k]
    return test_lst
        

