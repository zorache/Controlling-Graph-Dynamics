# Controlling Epidemic Diffusion with Temporal Graph and Graph Neural Network
CS599 Final Project


How can we leverage graph information to test efficiently and accurately the most high impact nodes to control epidemic diffusion? 


This project contains an implementation of the SEIR framework of epidemic spread, 2-hop heuristic method and supervised GNN as described in the ICML 2021 Paper, Controlling Graph Dynamics with Reinforcement Learning and Graph Neural Networks (Meirom et al. 2021). This project experiments on preferential attachment networks, the General Relativity and Quantum Cosmology Collaboration Network (Leskovec et al. 2007), and includes another heuristic method of testing every half incubation time. 

[Slides](https://docs.google.com/presentation/d/1h7MXzv8cilig4e3P40PhL22erHtSj7alUFHSK-jk8_4/edit?usp=sharing)
 

To train the GNN

`python train.py`

To test the GNN based on saved model 

`python test.py`

To test heuristic baselines

`python baseline.py`

