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
    
    