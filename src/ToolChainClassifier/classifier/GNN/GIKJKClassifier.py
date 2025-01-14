import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_add_pool, global_mean_pool

# class GINJK(torch.nn.Module):
#     def __init__(self, num_features, hidden, num_classes, dropout=0.0, l1_reg=0.0, l2_reg=0.0):
#         super(GINJK, self).__init__()
#         self.conv1 = GINConv(
            # torch.nn.Sequential(
            #     torch.nn.Linear(num_features, hidden),
            #     torch.nn.BatchNorm1d(hidden),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(hidden, hidden),
            #     torch.nn.BatchNorm1d(hidden),
            #     torch.nn.ReLU()))
#         self.conv2 = GINConv(
#             torch.nn.Sequential(
#                 torch.nn.Linear(hidden, hidden),
#                 torch.nn.BatchNorm1d(hidden),
#                 torch.nn.ReLU(),
#                 torch.nn.Linear(hidden, hidden),
#                 torch.nn.BatchNorm1d(hidden),
#                 torch.nn.ReLU()))
#         self.fc = torch.nn.Linear(hidden, num_classes)
#         self.dropout = dropout
#         self.l1_reg = l1_reg
#         self.l2_reg = l2_reg

#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = global_add_pool(x, batch)
#         x = self.fc(x)

#         # Add L1 and L2 regularization
#         if self.l1_reg > 0:
#             l1_loss = 0.0
#             for param in self.parameters():
#                 l1_loss += torch.norm(param, p=1)
#             x += self.l1_reg * l1_loss

#         if self.l2_reg > 0:
#             l2_loss = 0.0
#             for param in self.parameters():
#                 l2_loss += torch.norm(param, p=2)
#             x += 0.5 * self.l2_reg * l2_loss

#         return F.log_softmax(x, dim=-1)

class GINJK(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, num_layers=4):
        super(GINJK, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    GINConv(
                        torch.nn.Sequential(
                            torch.nn.Linear(num_features, hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden, hidden),
                        ), train_eps=False
                    )
                )
            else:
                self.convs.append(
                    GINConv(
                        torch.nn.Sequential(
                            torch.nn.Linear(hidden, hidden),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hidden, hidden),
                        )
                    )
                )
        self.fc = torch.nn.Linear(hidden * num_layers, num_classes)  # Adjust the output dimension

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)
        x = torch.cat(xs, dim=1)  # Concatenate representations from all layers
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
    
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
    #     super(GIN_JK, self).__init__()
    #     self.num_layers = num_layers
    #     self.convs = torch.nn.ModuleList()
    #     self.jump = JumpingKnowledge(mode='cat')

    #     for i in range(num_layers):
    #         if i == 0:
    #             self.convs.append(GINConv(torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU())))
    #         else:
    #             self.convs.append(GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU())))
    #     self.lin1 = torch.nn.Linear(hidden_channels * num_layers + in_channels, hidden_channels)
    #     self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    # def forward(self, data):
    #     x, edge_index, batch = data.x, data.edge_index, data.batch
    #     xs = []
    #     for i in range(self.num_layers):
    #         if i == 0:
    #             x = self.convs[i](x, edge_index)
    #         else:
    #             x = self.convs[i](x, edge_index)
    #         xs += [global_add_pool(x, batch)]
    #     x = self.jump(xs)
    #     x = torch.cat([x, global_add_pool(data.x, data.batch)], dim=1)
    #     x = F.relu(self.lin1(x))
    #     x = self.lin2(x)
    #     return F.log_softmax(x, dim=1)

# dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
# loader = DataLoader(dataset, batch_size=64, shuffle=True)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GINJK(dataset.num_node_features, 64, dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()

# def train():
#     model.train()
#     loss_all = 0
#     for data in loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.batch)
#         loss = criterion(out, data.y)
#         loss.backward()
#         loss_all += data.num_graphs * loss.item()
#         optimizer.step()
#     return loss_all / len(dataset)

# @torch.no_grad()
# def test():
#     model.eval()
#     correct = 0
#     for data in loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.batch)
#         pred = out.argmax(dim=1)
#         correct += pred.eq(data.y).sum().item()
#     return correct / len(dataset)

# for epoch in range(1, 201):
#     loss = train()
#     acc = test()
#     print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
