## Introduction
This is our implementation of our paper *GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks*.

**TL;DR**: A graph topology-oriented prompting framework.

**Abstract**:
Graph Neural Networks (GNNs) have revolutionized the field of graph learning by learning expressive graph representations from massive graph data. 
As a common pattern to train powerful GNNs, the "pre-training, adaptation" scheme first pre-trains GNNs over unlabeled graph data and subsequently adapts them to specific downstream tasks. 
In the adaptation phase, graph prompting is an effective strategy that modifies input graph data with learnable prompts while keeping pre-trained GNN models frozen. 
Typically, existing graph prompting studies mainly focus on feature-oriented methods that apply graph prompts to node features or hidden representations. 
However, these studies often achieve suboptimal performance, as they consistently overlook the potential of topology-oriented prompting, which adapts pre-trained GNNs by modifying the graph topology. 
In this study, we conduct a pioneering investigation of graph prompting in terms of graph topology. 
We propose the first Graph Topology-Oriented Prompting (GraphTOP) framework to effectively adapt pre-trained GNN models for downstream tasks. 
More specifically, we reformulate topology-oriented prompting as an edge rewiring problem within multi-hop local subgraphs and relax it into the continuous probability space through reparameterization while ensuring tight relaxation and preserving graph sparsity. 
Extensive experiments on five graph datasets under four pre-training strategies demonstrate that our proposed GraphTOP outshines six baselines on multiple node classification datasets.

## Dependencies
- torch==2.2.1  
- torch-geometric==2.5.2  
- torch-cluster==1.6.3   
- torch-scatter==2.1.2
- torch-sparse==0.6.18  
- numpy==1.26.1

## Usage
##### 1. Install dependencies
```
conda create --name GraphTOP -y python=3.9
conda activate GraphTOP
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.1 torch-geometric==2.5.2
pip install torch-cluster torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
```
##### 2. Run code
Example: run GraphTOP on PubMed with LP-GraphPrompt as the pre-training task
```
python run.py --dataset_name=PubMed --pretrain_task=EdgePredGraphPrompt
```

