import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden)))
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden)))
        self.fc = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
    
    # def __init__(self, num_features, hidden, num_classes):
    #     super(GIN, self).__init__()
    #     self.conv1 = GINConv(torch.nn.Sequential(torch.nn.Linear(num_features, hidden), torch.nn.ReLU()))
    #     self.conv2 = GINConv(torch.nn.Sequential(torch.nn.Linear(hidden, num_classes), torch.nn.ReLU()))
    #     self.pool = global_add_pool

    # def forward(self, data):
    #     x, edge_index, batch = data.x, data.edge_index, data.batch
    #     x = self.conv1(x, edge_index)
    #     x = F.relu(x)
    #     x = self.conv2(x, edge_index)
    #     x = self.pool(x, batch)
    #     return F.log_softmax(x, dim=1)



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GIN(dataset.num_node_features, 64, dataset.num_classes).to(device)
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
