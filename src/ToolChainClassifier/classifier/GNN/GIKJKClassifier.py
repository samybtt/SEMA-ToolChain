import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_add_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

class GINJK(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GINJK, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(in_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels))))
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_channels, hidden_channels),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_channels, hidden_channels))))
        self.jump = JumpingKnowledge(mode='cat')
        self.fc = torch.nn.Linear(num_layers * hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)
        x = self.jump(xs)
        x = global_add_pool(x, batch)
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
