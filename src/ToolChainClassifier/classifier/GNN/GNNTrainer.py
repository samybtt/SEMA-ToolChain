import torch
import torch.nn.functional as F
import progressbar
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score , f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import pandas as pd
import os
import glob
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
import numpy as np
import logging

from ..Classifier import Classifier
from .GraphSAGEClassifier import GraphSAGE
from .GINClassifier import GIN
from .GIKJKClassifier import GINJK
from .utils import gen_graph_data, read_gs_4_gnn, PyGDataset

CLASS_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY_CLASS = False # TODO
def collate_fn(data_list):
        print("AAAAAAAAAAA")
        dimensions = [max([d.x.size()[i] for d in data_list]) for i in range(len(data_list[0].x.size()))]
        for i in range(len(data_list)):
            data_list[i] = F.pad(data_list[i].x, [(0, dimensions[j] - data_list[i].size()[j]) for j in range(len(dimensions))])
        batched_data = Batch.from_data_list(data_list)
        return batched_data

class GNNTrainer(Classifier):
    def __init__(self, path, name, threshold=0.45,
                 families=['bancteian','delf','FeakerStealer','gandcrab','ircbot','lamer','nitol','RedLineStealer','sfone','sillyp2p','simbot','Sodinokibi','sytro','upatre','wabot','RemcosRAT'],
                 multi_gpu=1):
        super().__init__(path, name, threshold)

        self.multi_gpu = multi_gpu

        self.path = path
        self.name = name
        self.families = families
        self.mapping = self.read_mapping('mapping.txt')
        self.mapping_inv = self.read_mapping_inverse('mapping.txt')
        self.dataset = []
        self.label = []
        self.label_dict = {}
        self.fam_idx = []
        self.fam_dict = {}
        self.n_features = 0
        classes = set()
        for fname in glob.glob("{0}/*/*.gs".format(path)):
            dirname,name = self.get_label(fname)
            # self.data.append((fname,dirname))
            classes.add(dirname)
        self.classes = sorted(list(classes))
        self.clf = None
        self.y_pred = list()

        self.train_index, self.val_index = [],[]
        self.original_path = ""
        self.train_dataset, self.val_dataset, self.y_train, self.y_val = [],[],[],[]
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")


    def init_dataset(self, path):
        if path[-1] != "/":
            path += "/"
        self.log.info("Path: " + path)
        bar = progressbar.ProgressBar(max_value=len(self.families))
        bar.start()
        self.original_path = path
        self.dataset = []
        self.label = []
        for family in self.families:
            path = self.path + '/'  + self.original_path + family + '/'
            path = path.replace("ToolChainClassifier/","") # todo
            self.log.info("Subpath: " + path)
            if not os.path.isdir(path) :
                self.log.info("Dataset should be a folder containing malware classify by familly in subfolder")
                exit(-1)
            else:
                #filenames = glob.glob(path+'/SCDG_*') + glob.glob(path+'test/SCDG_*')
                filenames = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                if len(filenames) > 1 and family not in self.fam_idx :
                    self.fam_idx.append(family)
                    self.fam_dict[family] = len(self.fam_idx) - 1
                for file in filenames:
                    if file.endswith(".gs"):
                        edges, nodes, vertices, edge_labels = read_gs_4_gnn(file,self.mapping)
                        data = gen_graph_data(edges, nodes, vertices, edge_labels, self.fam_dict[family])
                        if len(data.edge_index.size()) > 1:
                            if len(nodes) > 1 and len(data.edge_index.size()) > 1:
                                self.dataset.append(data)
                            if BINARY_CLASS and len(nodes) > 1:
                                if family == 'clean':
                                    self.label.append(family)
                                else:
                                    self.label.append('malware')
                            else:
                                if len(nodes) > 1:
                                    self.label.append(family)
        # import pdb; pdb.set_trace()
        bar.finish()

    def split_dataset(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=24)
        for train, test in sss.split(self.dataset, self.label):
            self.train_index = train
            self.val_index = test
        for i in self.train_index:
            self.train_dataset.append(self.dataset[i])
            self.y_train.append(self.label[i])  
        for i in self.val_index:
            self.val_dataset.append(self.dataset[i])
            self.y_val.append(self.label[i])

    def get_label(self,fname):
        name = os.path.basename(fname)
        dirname = os.path.basename(os.path.dirname(fname))
        dname = dirname.split("_")[0]
        return dname, name
    
    def train(self, path, epoch=250, lr=0.01):
        self.init_dataset(path)

        self.log.info("Dataset len: " + str(len(self.dataset)))
        self.dataset_len = len(self.dataset)

        if self.dataset_len > 0:            
            self.split_dataset()
            dataset = self.train_dataset
            # import pdb; pdb.set_trace()

            # dataset = TUDataset(root='data/TUDataset', name='MUTAG')
            num_classes = len(np.unique(self.label))
            train_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

            # import pdb; pdb.set_trace()
            
            if self.name=="gin":
                self.clf = GIN(dataset[0].num_node_features, 64, num_classes)
            elif self.name=="sage":
                self.clf = GraphSAGE(dataset[0].num_node_features, 64, num_classes)
            elif self.name=="ginjk":
                self.clf = GINJK(dataset[0].num_node_features, 64, num_classes)

            optimizer = torch.optim.Adam(self.clf.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
            
            # import pdb; pdb.set_trace()

            for epoch in range(epoch):
                self.clf.train()
                # import pdb; pdb.set_trace()
                loss_all = 0
                for data in train_loader:
                    optimizer.zero_grad()
                    out = self.clf(data.x, data.edge_index, data.batch)
                    loss = criterion(out, data.y)
                    loss.backward()
                    loss_all += data.num_graphs * loss.item()
                    optimizer.step()
                if(epoch % 10 == 0):
                    print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_all / len(dataset)))
            self.log.info('--------------------FIT OK----------------')
        else:
            self.log.info("Dataset length should be > 0")
            exit(-1)

    @torch.no_grad()
    def classify(self,path=None):
        if path is None:
            self.clf.eval()
            val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)
            self.y_pred = list()
            correct = 0
            for data in val_loader:
                out = self.clf(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                # correct += int((pred == data.y).sum())
                correct += pred.eq(data.y).sum().item()
                for p in pred:
                    # print(p)
                    self.y_pred.append(self.fam_idx[p])
            print("AAAAAccuracy: " + str(correct/len(self.dataset)))
        else:
            # import pdb; pdb.set_trace()
            self.init_dataset(path)
            self.clf.eval()
            val_loader = DataLoader(self.dataset, batch_size=64, shuffle=False)
            self.y_pred = list()
            correct = 0
            
            for data in val_loader:
                out = self.clf(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                # correct += int((pred == data.y).sum())
                correct += pred.eq(data.y).sum().item()
                for p in pred:
                    # print(p)
                    self.y_pred.append(self.fam_idx[p])
            print("AAAAAccuracy: " + str(correct/len(self.dataset)))
            # import pdb; pdb.set_trace() 


    @torch.no_grad()
    def detection(self,path=None):
        if path is None:
            self.clf.eval()
            val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)
            self.y_pred = list()
            correct = 0
            # import pdb; pdb.set_trace() 
            for data in val_loader:
                out = self.clf(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                # correct += int((pred == data.y).sum())
                correct += pred.eq(data.y).sum().item()
                for p in pred:
                    # print(p)
                    self.y_pred.append(self.fam_idx[p])
            print("AAAAAccuracy: " + str(correct/len(self.dataset)))
        else:
            self.init_dataset(path)
            self.clf.eval()
            val_loader = DataLoader(self.dataset, batch_size=64, shuffle=False)
            self.y_pred = list()
            correct = 0
            # import pdb; pdb.set_trace() 
            for data in val_loader:
                out = self.clf(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                # correct += int((pred == data.y).sum())
                correct += pred.eq(data.y).sum().item()
                for p in pred:
                    # print(p)
                    self.y_pred.append(self.fam_idx[p])
            print("AAAAAccuracy: " + str(correct/len(self.dataset)))


    def get_stat_classifier(self):
        logging.basicConfig(level=logging.INFO)
        

        self.log.info("Accuracy %2.2f %%" %(accuracy_score(self.label, self.y_pred)*100))
        self.log.info("Precision %2.2f %%" %(precision_score(self.label, self.y_pred,average='weighted')*100))
        self.log.info("Recall %2.2f %%" %(recall_score(self.label, self.y_pred,average='weighted')*100))
        f_score = f1_score(self.label, self.y_pred,average='weighted')*100
        self.log.info("F1-score %2.2f %%" %(f_score))
    
        if BINARY_CLASS:
            conf = confusion_matrix(self.label,self.y_pred,labels=['clean','malware'])
            y_score1 = self.clf.predict_proba(self.K_val)[:,1]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(self.label, y_score1,pos_label='clean')
            plt.subplots(1, figsize=(10,10))
            plt.title('Receiver Operating Characteristic - DecisionTree')
            plt.plot(false_positive_rate1, true_positive_rate1)
            plt.plot([0, 1], ls="--")
            plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            print("OOOOOOHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
            # plt.savefig(self.original_path + "figure_binary.png")

        else:
            conf = confusion_matrix(self.label,self.y_pred,labels=self.fam_idx)

        list_name =[]
        for y in self.label:
            if y not in list_name:
                list_name.append(y)
        figsize = (10,7)
        fontsize=9
        if BINARY_CLASS:
            df_cm = pd.DataFrame(conf, index=['clean','malware'], columns=['clean','malware'],)
        else :
            df_cm = pd.DataFrame(conf, index=self.fam_idx, columns=self.fam_idx,)
        fig = plt.figure(figsize=figsize)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d",cbar=False)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        print("AAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        # plt.savefig(self.original_path + "figure.png")
        plt.savefig("/home/sambt/Desktop/figure.png")
        return f_score