from torch_geometric.data import Data
from torch_geometric.explain import Explainer, PGExplainer, GNNExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.explain.metric import fidelity
import networkx as nx
import matplotlib.pyplot as plt

# dataset = ...
# loader = DataLoader(dataset, batch_size=1, shuffle=True)

class GNNExplainability():
    def __init__(self,dataset,loader,model, output_path=None):
        self.dataset = dataset
        self.loader = loader
        self.model = model
        self.output_path= output_path

    def explain(self):
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='phenomenon',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='log_probs',
            ),
            threshold_config=dict(threshold_type='topk', value=20),
        )

        for i in range(len(self.dataset)):
            explanation = explainer(self.dataset[i].x, self.dataset[i].edge_index, target=self.dataset[i].y, batch=self.dataset[i].batch)
            print(explanation)
            # import pdb; pdb.set_trace()
            pred = explainer.get_prediction(self.dataset[i].x, self.dataset[i].edge_index, self.dataset[i].batch).argmax(dim=1).item()
            true_label = self.dataset[i].y.item()
            # fid_pm = fidelity(explainer, explanation)
            # print(fid_pm)
            if self.output_path is not None:
                explanation.visualize_graph(self.output_path+f'subgraph_{i}_{true_label}_{pred}.png', backend="graphviz")
                # explanation.visualize_feature_importance(self.output_path+f'features_subgraph_{i}.png', top_k=10)
            else:
                explanation.visualize_graph(f'subgraph_{i}_{true_label}_{pred}.png', backend="graphviz")
                # explanation.visualize_feature_importance(f'features_subgraph_{i}.png', top_k=10)

    # def explain(self):
    #     explainer = Explainer(
    #         model=self.model,
    #         algorithm=PGExplainer(epochs=30, lr=0.003),
    #         explanation_type='phenomenon',
    #         edge_mask_type='object',
    #         model_config=dict(
    #             mode='multiclass_classification',
    #             task_level='graph',
    #             return_type='raw',
    #         ),
    #         # Include only the top 10 most important edges:
    #         threshold_config=dict(threshold_type='topk', value=10),
    #     )

    #     # PGExplainer needs to be trained separately since it is a parametric
    #     # explainer i.e it uses a neural network to generate explanations:
    #     for epoch in range(30):
    #         for data in self.loader:
    #             # import pdb; pdb.set_trace()
    #             loss = explainer.algorithm.train(
    #                 epoch, self.model, data.x, data.edge_index, target=data.y, batch=data.batch)

    #     # Generate the explanation for a particular graph:
        # for i in range(len(self.dataset)):
        #     explanation = explainer(self.dataset[i].x, self.dataset[i].edge_index, target=self.dataset[i].y, batch=self.dataset[i].batch)
        #     print(explanation.edge_mask)
        #     import pdb; pdb.set_trace()
        #     # fid_pm = fidelity(explainer, explanation)
        #     # print(fid_pm)
            
        #     if self.output_path is not None:
        #         explanation.visualize_graph(self.output_path+f'subgraph_{i}.png', backend="networkx")
        #         # explanation.visualize_feature_importance(self.output_path+f'features_subgraph_{i}.png', top_k=10)
        #     else:
        #         explanation.visualize_graph(f'subgraph_{i}.png', backend="networkx")
        #         # explanation.visualize_feature_importance(f'features_subgraph_{i}.png', top_k=10)
