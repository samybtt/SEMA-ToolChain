import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import degree
from torch.nn import Linear


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, num_layers=4):
        super(GraphSAGE, self).__init__()
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden, hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden, hidden),
                    )))
        self.fc = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        # x = F.relu(self.fc(x))
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


    #     self.conv1 = GINConv(
    #         torch.nn.Sequential(
    #             torch.nn.Linear(num_features, hidden),
    #             torch.nn.ReLU(),
    #             torch.nn.BatchNorm1d(hidden),
    #             torch.nn.Linear(hidden, hidden),
    #         ), train_eps=False)
    #     self.convs = torch.nn.ModuleList()
    #     for i in range(num_layers - 1):
    #         self.convs.append(
    #             GINConv(
    #                 torch.nn.Sequential(
    #                     torch.nn.Linear(hidden, hidden),
    #                     torch.nn.ReLU(),
    #                     torch.nn.BatchNorm1d(hidden),
    #                     torch.nn.Linear(hidden, hidden),
    #                 )))
    #     self.fc = torch.nn.Linear(hidden, num_classes)

    # def forward(self, x, edge_index, batch):
    #     x = F.relu(self.conv1(x, edge_index))
    #     for conv in self.convs:
    #         x = F.relu(conv(x, edge_index))
    #     x = global_mean_pool(x, batch)
    #     x = F.relu(self.fc(x))
    #     # x = self.fc(x)
    #     return F.log_softmax(x, dim=-1)
    
        # self.conv1 = GINConv(
        #     torch.nn.Sequential(
        #         torch.nn.Linear(num_features, hidden),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(hidden, hidden)))
        # self.bn1 = torch.nn.BatchNorm1d(hidden)
        # self.conv2 = GINConv(
        #     torch.nn.Sequential(
        #         torch.nn.Linear(hidden, hidden),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(hidden, hidden)))
        # self.bn2 = torch.nn.BatchNorm1d(hidden)
        # self.fc = torch.nn.Linear(hidden, num_classes)
        # self.dropout = dropout
        # self.l1_reg = l1_reg
        # self.l2_reg = l2_reg

    # def forward(self, x, edge_index, batch):
    #     x = F.relu(self.conv1(x, edge_index))
    #     x = self.bn1(x)
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = F.relu(self.conv2(x, edge_index))
    #     x = self.bn2(x)
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = global_add_pool(x, batch)
    #     x = self.fc(x)
    #     if self.l1_reg > 0:
    #         l1_loss = torch.tensor(0.0, device=x.device)
    #         for param in self.parameters():
    #             l1_loss += torch.norm(param, p=1)
    #         x += self.l1_reg * l1_loss
    #     if self.l2_reg > 0:
    #         l2_loss = torch.tensor(0.0, device=x.device)
    #         for param in self.parameters():
    #             l2_loss += torch.norm(param, p=2)
    #         x += 0.5 * self.l2_reg * l2_loss
    #     return F.log_softmax(x, dim=-1)
    
    
    # def __init__(self, num_features, hidden, num_classes):
    #     super(GraphSAGE, self).__init__()
    #     self.conv1 = GATConv(num_features, hidden)
    #     self.conv2 = GATConv(hidden, num_classes)

    #     self.fc = torch.nn.Linear(hidden, num_classes)
    # def forward(self, x, edge_index, batch):
    #     x = F.elu(self.conv1(x, edge_index))
    #     x = self.conv2(x, edge_index)
    #     x = global_max_pool(x, batch) # global max pooling
    #     self.fc(x)
    #     return F.log_softmax(x, dim=-1)

        
    # def forward(self, x, edge_index, batch):
    #     # Compute normalization factor for each node's neighbors
    #     row, col = edge_index
    #     deg = degree(col, x.size(0), dtype=x.dtype)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    #     # Apply GraphSAGE convolutional layers
    #     x = self.conv1(x, edge_index, norm)
    #     x = F.relu(x)
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.conv2(x, edge_index, norm)

    #     # Sum node features for each graph in the batch
    #     out = torch.zeros((edge_index.max().item() + 1, x.size(1)), dtype=x.dtype, device=x.device)
    #     out[edge_index[0]] += x[edge_index[1]]
    #     return out.sum(dim=0, keepdim=True)
    
    # def forward(self, x, edge_index, batch):
    #     x = F.relu(self.conv1(x, edge_index))
    #     x = F.relu(self.conv2(x, edge_index))
    #     x = self.lin(x)
    #     return F.log_softmax(x, dim=-1)

# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GraphSAGE(dataset.num_features, 128, dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()

# for epoch in range(1, 201):
#     model.train()
#     loss_all = 0
#     for data in loader:
#         data = data
#     print(f'Epoch {epoch:03d}, Loss: {loss_all:.4f}')
