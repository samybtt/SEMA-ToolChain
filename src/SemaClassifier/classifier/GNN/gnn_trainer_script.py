import time

from matplotlib import pyplot as plt
import GNN_script
import argparse
import os
import sys
import torch
import numpy as np
import flwr as fl
from utils import read_mapping, read_mapping_inverse
from collections import OrderedDict
from typing import Dict, List, Tuple
import copy

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score , f1_score, balanced_accuracy_score

from GINJKFlagClassifier import GINJKFlag
from GINClassifier import GIN
from GINJKClassifier import GINJK
from RGINClassifier import R_GINJK
from RGINJKClassifier import RanGINJK
import sys
sys.path.append("./SemaClassifier/classifier/")
from SVM.SVMClassifier import SVMClassifier
from SVM.SVMWLClassifier import SVMWLClassifier

from torch_geometric.loader import DataLoader

import pandas as pd
import seaborn as sns


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE_TEST=32

fam_idx = {}

def get_datasets(dataset, trn_idx, tst_idx):
    train_dataset = []
    test_dataset = []
    y_train = []
    y_test = []
    for i in trn_idx:
        train_dataset.append(dataset[i])
        y_train.append(dataset[i].y.item())
    for i in tst_idx:
        test_dataset.append(dataset[i])
        y_test.append(dataset[i].y.item())
    return train_dataset, y_train, test_dataset, y_test

def get_datasets_wl(dataset, trn_idx, tst_idx, label):
    train_dataset = []
    test_dataset = []
    y_train = []
    y_test = []
    for i in trn_idx:
        train_dataset.append(dataset[i])
        y_train.append(label[i])
    for i in tst_idx:
        test_dataset.append(dataset[i])
        y_test.append(label[i])
    return train_dataset, y_train, test_dataset, y_test

def one_epoch_train_vanilla(model, train_loader, val_loader, device, optimizer, criterion):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        train_loss += loss.item() * data.num_graphs
        optimizer.step()
        train_total += data.num_graphs
        train_correct += (output.argmax(1) == data.y).sum().item()
    train_loss /= train_total
    train_acc = train_correct / train_total
    # val_acc, val_loss, _ = test(model, val_loader, BATCH_SIZE_TEST, device)
    return model, train_acc, train_loss, 0, 0 #val_acc, val_loss

def one_epoch_train_flag(model, train_loader, val_loader, device, optimizer, criterion, step_size, m):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # import pdb; pdb.set_trace()

        perturb = torch.FloatTensor(data.x.shape[0], model.hidden).uniform_(-step_size, step_size).to(device)
        perturb.requires_grad_()

        output = model(data.x, data.edge_index, data.edge_attr, data.batch, perturb)
        loss = criterion(output, data.y)
        loss /= m

        for _ in range(m-1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            output = model(data.x, data.edge_index, data.edge_attr, data.batch, perturb)
            loss = criterion(output, data.y)
            loss /= m

        loss.backward()
        train_loss += loss.item() * data.num_graphs
        optimizer.step()
        train_total += data.num_graphs
        train_correct += (output.argmax(1) == data.y).sum().item()
    train_loss /= train_total
    train_acc = train_correct / train_total
    # val_acc, val_loss, _ = test(model, val_loader, BATCH_SIZE_TEST, device)
    return model, train_acc, train_loss, 0, 0 #val_acc, val_loss

def train(model, train_dataset, val_dataset, batch_size, device, epochs, step_size=8e-3, m=3, flag=False, lr=0.001):
    print(f"Training with flag: {flag}, step_size: {step_size}, m: {m}, lr: {lr}")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                            T_max = 42, # Maximum number of iterations.
                            eta_min = 1e-4) # Minimum learning rate.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        if flag:
            model, train_acc, train_loss, val_acc, val_loss = one_epoch_train_flag(model, train_loader, val_loader, device, optimizer, criterion, step_size, m)
        else:
            model, train_acc, train_loss, val_acc, val_loss = one_epoch_train_vanilla(model, train_loader, val_loader, device, optimizer, criterion)
        scheduler.step()
        # GNN_script.cprint(f"Epoch {epoch+1}: Train acc: {train_acc:.4%} | Train loss: {train_loss:.4} | Test accuracy: {val_acc:.4%} | Test loss: {val_loss:.4}", 1)
        GNN_script.cprint(f"Epoch {epoch+1}: Train acc: {train_acc:.4%} | Train loss: {train_loss:.4}", 1)
    return model 

def test(model, test_loader , batch_size, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    y_pred = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.inference_mode():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(output, data.y).item()
            test_loss += loss * data.num_graphs
            test_total += data.num_graphs
            pred = output.argmax(dim=1)
            test_correct += pred.eq(data.y).sum().item()
            for p in pred:
                y_pred.append(p.item())
    test_loss /= test_total
    test_acc = test_correct / test_total
    return test_acc, test_loss, y_pred

def computre_metrics(y_true, y_pred, fam_idx):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return acc, prec, rec, f1, bal_acc

def plot_confusion_matrix(y_true, y_pred, fam_idx, model_name):
    # plot confusion matrix
    if type(y_true[0]) != str:  
        y_true_label = [fam_idx[i] for i in y_true]
        y_pred_label = [fam_idx[i] for i in y_pred]
    else:
        y_true_label = y_true
        y_pred_label = y_pred

    # import pdb; pdb.set_trace()
    cm = confusion_matrix(y_true_label, y_pred_label, labels=np.unique(fam_idx))
    print(cm)

    df_cm = pd.DataFrame(cm, index = np.unique(fam_idx),
                    columns = np.unique(fam_idx))
    plt.figure(figsize = (10,7))
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d",cbar=False)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"Confusion matrix for {model_name}")
    plt.savefig(f"confusion_matrix_{model_name}_1.png")
    plt.show()

def tune_parameters_ginjk(full_train_dataset, train_dataset, val_dataset, y_val, test_dataset, y_test, num_classes):
    hidden = [32, 64, 128]
    num_layers = [2, 3, 4, 5, 6]
    lr = [0.001, 0.0001]
    batch_sizes = [1, 8, 16, 32, 64]
    best_params = {}
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    best_bal_acc = 0
    best_loss = float('inf')
    best_fscore = 0
    
    for h in hidden:
        for l in num_layers:
            for r in lr:
                current_params = {}
                current_params["hidden"] = h
                current_params["layers"] = l
                current_params["lr"] = r
                current_params["batch_size"] = batch_size
                print(f"Hidden: {h}, Layers: {l}, LR: {r}")
                model = GINJK(train_dataset[0].num_node_features, h, num_classes, l).to(DEVICE)
                start = time.time()
                model = train(model, train_dataset, val_dataset, batch_size, DEVICE, epochs)
                end = time.time()
                trn_time = end - start
                print(f"Training time: {trn_time}")
                current_params["training_time"] = trn_time
                start = time.time()
                accuracy, loss, y_pred = test(model, val_loader, batch_size, DEVICE)
                end = time.time()
                print(f"Testing time: {end - start}")
                current_params["loss"] = loss
                current_params["acc"], current_params["prec"], current_params["rec"], current_params["f1"], current_params["bal_acc"] = computre_metrics(y_val, y_pred)
                if current_params["bal_acc"] > best_bal_acc:
                    best_bal_acc = current_params["bal_acc"]
                    best_loss = loss
                    best_fscore = current_params["f1"]
                    best_params["hidden"] = h
                    best_params["layers"] = l
                    best_params["lr"] = r
                    best_params["batch_size"] = batch_size
                    best_params["acc"] = current_params["acc"]
                    best_params["prec"] = current_params["prec"]
                    best_params["rec"] = current_params["rec"]
                    best_params["f1"] = current_params["f1"]
                    best_params["bal_acc"] = current_params["bal_acc"]
                    best_params["loss"] = loss
                print("Current:")
                print(current_params)
                print("Best:")
                print(best_params)
                write_stats_to_tmp_csv(current_params, "ginjk")
    # return best_params
    # Evaluate best model
    model = GINJKFlag(train_dataset[0].num_node_features, best_params["hidden"], num_classes, best_params["layers"]).to(DEVICE)
    # tain and get training time:
    start = time.time()
    model = train(model, full_train_dataset, test_dataset, best_params["batch_size"], DEVICE, epochs, best_params["step_size"], best_params["m"], best_params["flag"], best_params["lr"])
    end = time.time()
    accuracy, loss, y_pred = test(model, test_loader, best_params["batch_size"], DEVICE)
    final_acc, final_prec, final_rec, final_f1, final_bal_acc = computre_metrics(y_test, y_pred)
    results = {}
    results["final_acc"] = final_acc
    results["final_prec"] = final_prec
    results["final_rec"] = final_rec
    results["final_f1"] = final_f1
    results["final_bal_acc"] = final_bal_acc
    results["final_loss"] = loss
    results["best_params"] = best_params
    results["training_time"] = end - start
    return results

def tune_parameters_fginjk(full_train_dataset, train_dataset, val_dataset, y_val, test_dataset, y_test, num_classes):
    hidden = [32, 64, 128]
    num_layers = [2, 3, 4, 5]
    lr = [0.001, 0.0001]
    batch_sizes = [1, 8, 16, 32, 64]
    flag = [True, False]
    step_size = [8e-3, 5e-3, 1e-3]
    m_steps = [3, 5, 7]
    best_params = {}
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    best_bal_acc = 0
    best_loss = float('inf')
    best_fscore = 0
    
    for h in hidden:
        for l in num_layers:
            for r in lr:
                for fg in flag:
                    if fg:
                        for m in m_steps:
                            for step in step_size:
                                current_params = {}
                                current_params["hidden"] = h
                                current_params["layers"] = l
                                current_params["lr"] = r
                                current_params["batch_size"] = batch_size
                                current_params["flag"] = fg
                                current_params["step_size"] = step
                                current_params["m"] = m

                                print(f"Hidden: {h}, Layers: {l}, LR: {r}, FLAG: {fg}, M: {m}, Step: {step}")
                                model = GINJKFlag(train_dataset[0].num_node_features, h, num_classes, l).to(DEVICE)
                                start = time.time()
                                model = train(model, train_dataset, val_dataset, batch_size, DEVICE, epochs, step, m, fg)
                                end = time.time()
                                trn_time = end - start
                                print(f"Training time: {trn_time}")
                                current_params["training_time"] = trn_time
                                start = time.time()
                                accuracy, loss, y_pred = test(model, val_loader, batch_size, DEVICE)
                                end = time.time()
                                print(f"Testing time: {end - start}")
                                current_params["loss"] = loss
                                current_params["acc"], current_params["prec"], current_params["rec"], current_params["f1"], current_params["bal_acc"] = computre_metrics(y_val, y_pred)
                                if current_params["bal_acc"] > best_bal_acc:
                                    best_bal_acc = current_params["bal_acc"]
                                    best_loss = loss
                                    best_fscore = current_params["f1"]
                                    best_params["hidden"] = h
                                    best_params["layers"] = l
                                    best_params["lr"] = r
                                    best_params["batch_size"] = batch_size
                                    best_params["acc"] = current_params["acc"]
                                    best_params["prec"] = current_params["prec"]
                                    best_params["rec"] = current_params["rec"]
                                    best_params["f1"] = current_params["f1"]
                                    best_params["bal_acc"] = current_params["bal_acc"]
                                    best_params["loss"] = loss
                                    best_params["flag"] = fg
                                    best_params["step_size"] = step
                                    best_params["m"] = m
                                print("Current:")
                                print(current_params)
                                print("Best:")
                                print(best_params)
                                write_stats_to_tmp_csv(current_params, "fginjk")
                    else:
                        current_params = {}
                        current_params["hidden"] = h
                        current_params["layers"] = l
                        current_params["lr"] = r
                        current_params["batch_size"] = batch_size
                        current_params["flag"] = fg
                        current_params["step_size"] = -1
                        current_params["m"] = -1
                        print(f"Hidden: {h}, Layers: {l}, LR: {r}, FLAG: {fg}")
                        model = GINJKFlag(train_dataset[0].num_node_features, h, num_classes, l).to(DEVICE)
                        start = time.time()
                        model = train(model, train_dataset, val_dataset, batch_size, DEVICE, epochs, flag=fg)
                        end = time.time()
                        trn_time = end - start
                        print(f"Training time: {trn_time}")
                        current_params["training_time"] = trn_time
                        start = time.time()
                        accuracy, loss, y_pred = test(model, val_loader, batch_size, DEVICE)
                        end = time.time()
                        print(f"Testing time: {end - start}")
                        current_params["loss"] = loss
                        current_params["acc"], current_params["prec"], current_params["rec"], current_params["f1"], current_params["bal_acc"] = computre_metrics(y_val, y_pred)
                        if current_params["bal_acc"] > best_bal_acc:
                            best_bal_acc = current_params["bal_acc"]
                            best_loss = loss
                            best_fscore = current_params["f1"]
                            best_params["hidden"] = h
                            best_params["layers"] = l
                            best_params["lr"] = r
                            best_params["batch_size"] = batch_size
                            best_params["acc"] = current_params["acc"]
                            best_params["prec"] = current_params["prec"]
                            best_params["rec"] = current_params["rec"]
                            best_params["f1"] = current_params["f1"]
                            best_params["bal_acc"] = current_params["bal_acc"]
                            best_params["loss"] = loss
                            best_params["flag"] = fg
                            best_params["step_size"] = -1
                            best_params["m"] = -1
                        print("Current:")
                        print(current_params)
                        print("Best:")
                        print(best_params)
                        write_stats_to_tmp_csv(current_params, "fginjk")
    # return best_params
    # Evaluate best model
    model = GINJKFlag(train_dataset[0].num_node_features, best_params["hidden"], num_classes, best_params["layers"]).to(DEVICE)
    # tain and get training time:
    start = time.time()
    model = train(model, full_train_dataset, test_dataset, best_params["batch_size"], DEVICE, epochs, best_params["step_size"], best_params["m"], best_params["flag"])
    end = time.time()
    accuracy, loss, y_pred = test(model, test_loader, best_params["batch_size"], DEVICE)
    final_acc, final_prec, final_rec, final_f1, final_bal_acc = computre_metrics(y_test, y_pred)
    results = {}
    results["final_acc"] = final_acc
    results["final_prec"] = final_prec
    results["final_rec"] = final_rec
    results["final_f1"] = final_f1
    results["final_bal_acc"] = final_bal_acc
    results["final_loss"] = loss
    results["best_params"] = best_params
    results["training_time"] = end - start
    return results

def tune_parameters_rdginjk(full_train_dataset, train_dataset, val_dataset, y_val, test_dataset, y_test, num_classes):
    rand_graph = ["ER"]
    hidden = [32, 64, 128]
    num_layers = [2, 4, 6, 8]

    

def write_stats_to_csv(results, clf_model):
    # Write stats and params in csv file
    if not os.path.isFile("stats.csv"):
        with open("stats.csv", "w") as f:
            f.write("model,acc,prec,rec,f1,bal_acc,loss,hidden,layers,lr,batch_size,flag,step_size,m,time\n")
    
    with open("stats.csv", "a") as f:
        f.write(f"{clf_model},{results['final_acc']},{results['final_prec']},{results['final_rec']},{results['final_f1']},{results['final_bal_acc']},{results['final_loss']},{results['best_params']['hidden']},{results['best_params']['layers']},{results['best_params']['lr']},{results['best_params']['batch_size']},{results['best_params']['flag']},{results['best_params']['step_size']},{results['best_params']['m']},{results['training_time']}\n")

def write_stats_to_tmp_csv(results, clf_model):
    # Write stats and params in csv file
    if not os.path.isFile("tmp_stats.csv"):
        with open("tmp_stats.csv", "w") as f:
            f.write("model,acc,prec,rec,f1,bal_acc,loss,hidden,layers,lr,batch_size,flag,step_size,m,time\n")
    
    with open("tmp_stats.csv", "a") as f:
        f.write(f"{clf_model},{results['acc']},{results['prec']},{results['rec']},{results['f1']},{results['bal_acc']},{results['loss']},{results['hidden']},{results['layers']},{results['lr']},{results['batch_size']},{results['flag']},{results['step_size']},{results['m']},{results['training_time']}\n")

def compare_models():
    pass

def init_all_datasets(path, families, mapping, reversed_mapping):

    id = 1

    # PyG dataset
    dataset, label, fam_idx, fam_dict, dataset_wl = GNN_script.init_dataset(path, families, reversed_mapping, [], {}, False)
    train_idx, test_idx = GNN_script.split_dataset_indexes(dataset, label)
    full_train_dataset,y_full_train, test_dataset, y_test = get_datasets(dataset, train_idx, test_idx)
    GNN_script.cprint(f"GNN {id} : datasets length, {len(dataset)}, {len(full_train_dataset)}, {len(test_dataset)}",id)
    # Validation dataset
    trn_idx, val_idx = GNN_script.split_dataset_indexes(full_train_dataset, y_full_train)
    train_dataset, y_train, val_dataset, y_val = get_datasets(full_train_dataset, trn_idx, val_idx)

    # WL dataset
    wl_full_train_dataset,wl_y_full_train, wl_test_dataset,wl_y_test = get_datasets_wl(dataset_wl, train_idx, test_idx, label)
    GNN_script.cprint(f"WL {id} : datasets length, {len(dataset_wl)}, {len(wl_full_train_dataset)} {len(wl_test_dataset)}",id)

    return full_train_dataset, y_full_train, test_dataset, y_test, train_dataset, y_train, val_dataset, y_val, wl_full_train_dataset, wl_y_full_train, wl_test_dataset, wl_y_test, label, fam_idx

def main(batch_size, hidden, num_layers, drop_ratio, residual, rand_graph, flag, step_size, m, epochs, net_linear, drop_path_p, edge_p, clf_model, tune, lr, ds_path):
    id = 1
    #Dataset Loading
    # families = ["berbew","sillyp2p","benjamin","small","mira","upatre","wabot"]
    # families = ['delf','FeakerStealer','gandcrab','ircbot','lamer','nitol','RedLineStealer','sfone','sillyp2p','sytro','wabot','RemcosRAT']
    families = ['delf','FeakerStealer','ircbot','lamer','nitol','RedLineStealer','sillyp2p','sytro','wabot','RemcosRAT']
    # families = ['delf','FeakerStealer','gandcrab','ircbot','lamer','nitol','RedLineStealer','sfone','sillyp2p','sytro','wabot','RemcosRAT','simbot','bancteian']
    # families = ["cleanware", "malware"]
    mapping = read_mapping("./mapping.txt")
    reversed_mapping = read_mapping_inverse("./mapping.txt")

    full_train_dataset, y_full_train, test_dataset, y_test, train_dataset, y_train, val_dataset, y_val, wl_full_train_dataset, wl_y_full_train, wl_test_dataset, wl_y_test, label, fam_idx = init_all_datasets(ds_path, families, mapping, reversed_mapping)

    num_classes = len(families)

    if not tune:
        #Model
        if clf_model == "fginjk":
            model = GINJKFlag(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
        elif clf_model == "ginjk":
            model = GINJK(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers).to(DEVICE)
        elif clf_model == "gin":
            model = GIN(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers).to(DEVICE)
        elif clf_model == "rdginjk":
            model = RanGINJK(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers,
            graph_model=rand_graph, drop_ratio=drop_ratio, residual=residual, net_linear=net_linear, drop_path_p=drop_path_p, edge_p=edge_p).to(DEVICE)
        elif clf_model == "rgin":
            model = R_GINJK(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
        elif clf_model == "wl":
            model = SVMWLClassifier("./databases/examples_samy/BODMAS/01", 0.45, families)
            model.train(dataset=wl_full_train_dataset, label=wl_y_full_train)
            wl_y_pred = model.classify(dataset=wl_test_dataset)

            wl_acc, wl_prec, wl_rec, wl_f1, wl_bal_acc = computre_metrics(wl_y_test, wl_y_pred, label)
            print()
            GNN_script.cprint("--------------------------------------------------",id)
            GNN_script.cprint(f"WL kernel Test accuracy: {wl_acc}",id)
            GNN_script.cprint(f"WL kernel Test balanced accuracy: {wl_bal_acc}",id)
            GNN_script.cprint(f"WL kernel Test precision: {wl_prec}",id)
            GNN_script.cprint(f"WL kernel Test recall: {wl_rec}",id)
            GNN_script.cprint(f"WL kernel Test f1: {wl_f1}",id)
            print()
            GNN_script.cprint("--------------------------------------------------",id)
            plot_confusion_matrix(wl_y_test, wl_y_pred, fam_idx, model_name="WL")
            return
        else:
            print("Invalid GNN model")
            return
        # Train model
        model = train(model, full_train_dataset, val_dataset, batch_size, DEVICE, epochs, step_size, m, flag, lr)

        # Test modelþ
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        accuracy, loss, y_pred = test(model, test_loader, batch_size, DEVICE)
        GNN_script.cprint(f"GNN: Evaluation accuracy & loss, {accuracy:%}, {loss}",id)
        # Compute metrics
        acc, prec, rec, f1, bal_acc = computre_metrics(y_test, y_pred, fam_idx)
        GNN_script.cprint("--------------------------------------------------",id)
        GNN_script.cprint(f"GNN: Test accuracy: {acc}",id)
        GNN_script.cprint(f"GNN: Test balanced accuracy: {bal_acc}",id)
        GNN_script.cprint(f"GNN: Test precision: {prec}",id)
        GNN_script.cprint(f"GNN: Test recall: {rec}",id)
        GNN_script.cprint(f"GNN: Test f1: {f1}",id)
        GNN_script.cprint("--------------------------------------------------",id)
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, fam_idx, model_name=clf_model)
    else:
        if clf_model == 'fginjk':
            GNN_script.cprint("Tuning parameters for fginjk",id)
            results = tune_parameters_fginjk(full_train_dataset, train_dataset, val_dataset, y_val, test_dataset, y_test, num_classes)
            write_stats_to_csv(results, clf_model)
        elif clf_model == 'ginjk':
            GNN_script.cprint("Tuning parameters for ginjk",id)
            results = tune_parameters_ginjk(full_train_dataset, train_dataset, val_dataset, y_val, test_dataset, y_test, num_classes)
            write_stats_to_csv(results, clf_model)
        else:
            print("Not implemented yet")
        


    
if __name__ == "__main__":
    print("Hello World")

    # Parse arguments:
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--num_layers', type=int, default=0, help='Number of GNN layers.')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='Dropout ratio (1 - keep probability).')
    parser.add_argument('--residual', type=bool, default=False, help='Whether to add residual connections.')
    parser.add_argument('--rand_graph', type=str, default='NA', help='Random graph model for randomly wired GNN.')
    parser.add_argument('--flag', action="store_true", help='Whether to use FLAG.')
    parser.add_argument('--step_size', type=float, default=8e-3 , help='Step size for FLAG.')
    parser.add_argument('--m', type=int, default=3, help='Ascent steps for FLAG.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs per iteration.')
    parser.add_argument('--net_linear', action="store_true", default=False, help='Whether to use linear layer.')
    parser.add_argument('--drop_path_p', type=float, default=0.01, help='Drop path probability.')
    parser.add_argument('--edge_p', type=float, default=0.6, help='Edge density in random graph.')
    parser.add_argument('--clf_model', type=str, default='fginjk', help='Which GNN to use.')
    parser.add_argument('--tune', action="store_true", help='Whether to tune parameters.')

    args = parser.parse_args()
    print(args)

    # Init variables according to arguments
    hidden = args.hidden
    batch_size = args.batch_size
    num_layers = args.num_layers
    drop_ratio = args.drop_ratio
    residual = args.residual
    rand_graph = args.rand_graph
    flag = args.flag
    step_size = args.step_size
    m = args.m
    lr = args.lr
    epochs = args.epochs
    net_linear = args.net_linear
    drop_path_p = args.drop_path_p
    edge_p = args.edge_p
    clf_model = args.clf_model
    tune = args.tune

    # ds_path = "./databases/examples_samy/BODMAS/01"
    # ds_path = "./databases/examples_samy/gs"
    ds_path = "./databases/examples_samy/out_serena/12/gs"
    # ds_path = "./databases/examples_samy/BODMAS/wselect3_01"
    # ds_path = "./databases/examples_samy/BODMAS/detection/cdfs_01"
    # ds_path = "./databases/examples_samy/big_dataset/merged/alldata/CDFS_b"
    # ds_path = "./databases/examples_samy/big_dataset/merged/alldata/WSELECTSET2_b"


    main(batch_size, hidden, num_layers, drop_ratio, residual, rand_graph, flag, step_size, m, epochs, net_linear, drop_path_p, edge_p, clf_model, tune, lr, ds_path)
    