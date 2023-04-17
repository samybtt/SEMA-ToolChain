import json
import pickle
import os
from glob import glob
import numpy as np
import progressbar
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.data import Batch
# from grakel import Graph
from torch.utils.data import Dataset

class PyGDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# Process the graph data and convert to a PyG Data object.
def gen_graph_data(edges, nodes, vertices, edge_labels, label):
    edges = list(edges)
    x = torch.tensor([torch.tensor(nodes[v]) for v in vertices])
    x = x.unsqueeze(-1)
    # import pdb; pdb.set_trace()
    edge_attr = torch.tensor([edge_labels[e] for e in edges])
    num_nodes = len(vertices)
    # edge_index=torch.tensor(edges).t().contiguous()
    y = torch.tensor([label])
    edge_index=torch.tensor(edges, dtype=torch.long)
    # print("edges 1 : ", edge_index.size())

    if (len(edge_index.size()) == 2):
        edge_index = edge_index.transpose(0, 1)
    # import pdb; pdb.set_trace()
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)
    # print(data)
    return data


def read_gs_4_gnn(path, lonely=True):
    f = open(path,'r')
    vertices = {}
    nodes = {}
    edges = {}
    edge_labels = {}
    c_edges = 1
    for line in f:
        if line.startswith("t"):
            pass
        if line.startswith("v"):
            sp = line.split(" ")
            v = int(sp[1])
            vertices[v] = []
            v_label = int(sp[2])
            # nodes[v] = mapping[v_label] 
            nodes[v] = v_label
        if line.startswith("e"):
            #self.log.info(line)
            sp = line.split(" ")
            v1 = int(sp[1])
            v2 = int(sp[2])
            edges[tuple((v1,v2))] = 1
            edge_labels[tuple((v1,v2))] = int(sp[3].replace('\n',''))
            c_edges = c_edges + 1
            vertices[v1].append(v2)
            vertices[v2].append(v1)
    
    if not lonely:
        #STUFF below to delete lonely nodes
        de = []
        count = 0
        vertices_ok = {}
        nodes_ok = {}
        map_clean = {}
        # find index of lonely node
        for key in vertices:
            if not vertices[key]:
                de.append(key)
            else:
                map_clean[key] = count
                count = count +1
        #delete them
        for key in de:
            del vertices[key]

        for key in vertices:
            local_dic = {}
            for v in vertices[key]:
                local_dic[map_clean[v]] = 1.0
            
            #self.log.info(local_dic)
            vertices_ok[map_clean[key]] = local_dic
            nodes_ok[map_clean[key]] = nodes[key]

        # if len(vertices_ok) <= 1:
        #     self.log.info(vertices_ok)

        # G = Graph(vertices_ok,node_labels=nodes_ok,edge_labels=edge_labels)
        G = edges, nodes_ok, vertices_ok, edge_labels
        # import pdb; pdb.set_trace()
    else:
        # G = Graph(vertices,node_labels=nodes,edge_labels=edge_labels)
        G = edges, nodes, vertices, edge_labels
        # import pdb; pdb.set_trace()
    f.close()
    return G

def read_mapping(path):
    map_file = open(path,'r')
    mapping = {}
    for line in map_file:
        tab = line.split('\n')[0].split(' ')
        mapping[int(tab[0])] = tab[1]
    map_file.close()
    return mapping


# read_gs_4_gnn('../../../out.gs', "clean") 