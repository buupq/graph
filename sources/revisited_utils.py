# visualizing training and validating results
from typing import Dict
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.utils import degree
from collections import Counter

# function to visualize the node degree
def visualize_node_degree(data: torch_geometric.data.data.Data):
    """visualize the node degree"""
    # node degree array
    degrees = degree(data.edge_index[0]).numpy()
    
    # count the number of node for each degree
    count_degree = Counter(degrees)
    
    # bar plot
    plt.bar(count_degree.keys(), count_degree.values())
    plt.xlabel("Node degree")
    plt.ylabel('Number of nodes')
    

def visualize_results(results:Dict):
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # plot train and validation loss
    ax[0].plot(results['epoch'], results['train_loss'], label='train_loss')
    ax[0].plot(results['epoch'], results['val_loss'], label='val_loss')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].set_title('TRAIN-VAL LOSS')
    ax[0].legend()

    # plot train and validation accuracy
    ax[1].plot(results['epoch'], results['train_acc'], label='train_acc')
    ax[1].plot(results['epoch'], results['val_acc'], label='val_acc')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('acc')
    ax[1].set_title('TRAIN-VAL ACC')
    ax[1].legend()

