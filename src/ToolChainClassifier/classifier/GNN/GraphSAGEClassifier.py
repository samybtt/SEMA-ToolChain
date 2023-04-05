import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import degree
from torch.nn import Linear


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                #  path,name,
                 threshold=0.45, 
                 families=['bancteian','delf','FeakerStealer','gandcrab','ircbot','lamer','nitol','RedLineStealer','sfone','sillyp2p','simbot','Sodinokibi','sytro','upatre','wabot','RemcosRAT']):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        import pdb; pdb.set_trace()
        x = x.long()
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # global max pooling

        return F.log_softmax(x, dim=1)
        
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
