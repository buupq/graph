# random walk function
import random
import networkx as nx

def random_walk(G: nx.Graph, start: int, length: int):
    
    """From `start` node, find `length` connected nodes"""
    
    # convert start node to string
    walk = [start]
    
    for i in range(length):
        neighbors = list(G.neighbors(start))
        next_node = random.choice(neighbors)
        walk.append(next_node)
        start = next_node
    
    return walk

import random
import networkx as nx
import numpy as np


def next_node(G: nx.Graph, previous: int, current: int, p: int=1, q: int=1):
    
    """get next node based on its probabilities"""
    
    # raw node probabilities
    alphas = []
    
    # get neighboring nodes
    neighbors = list(G.neighbors(current))
    
    # compute raw node probabilities
    for neighbor in neighbors:
        # distance = 0: probability to return to previous node
        if neighbor == previous:
            alpha = 1/p
        # distance = 1: probability to visit a local node
        elif G.has_edge(neighbor, previous):
            alpha = 1
        # distance = 2: probability to explore an unknown node
        else:
            alpha = 1/q
        # add raw probability to list
        alphas.append(alpha)
        
    # normalize the probabilities
    probs = [alpha / sum(alphas) for alpha in alphas]

    # randomly select a new node based on the transition probabilities
    next = random.choices(neighbors, weights=probs, k=1)[0]

    # return next node
    return next


def bias_walk(G: nx.Graph, start: int, length:int, p: int=1, q: int=1):

    """bias walk based on node probabilities"""
    
    # initialize the walk list
    walk = [start]
    
    for i in range(length):
        current = walk[-1]
        previous = walk[-2] if len(walk) > 1 else None
        next = next_node(G=G, previous=previous, current=current, p=p, q=q)
        walk.append(next)
        
    return walk