import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv


class GlobalScheduler(nn.Module):
    pass


class GlobalCritic(nn.Module):
    def __init__(self, meta_data, state_size, num_nodes, embed_dim, num_heads, num_HGT_layers, num_MLP_layers):
        super(GlobalCritic, self).__init__()
        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_MLP_layers = num_MLP_layers

        self.num_cranes = self.num_nodes["crane"]
        self.num_locations = self.num_nodes["machine"] + self.num_nodes["buffer"] + self.num_nodes["output"]

        self.conv = nn.ModuleList()
        for i in range(self.num_HGT_layers):
            if i == 0:
                self.conv.append(HGTConv(self.state_size, embed_dim, meta_data, heads=num_heads))
            else:
                self.conv.append(HGTConv(embed_dim, embed_dim, meta_data, heads=num_heads))

        self.mlp = nn.ModuleList()
        for i in range(num_MLP_layers):
            if i == 0:
                self.mlp.append(nn.Linear(embed_dim * 2, embed_dim))
            elif i < num_MLP_layers - 1:
                self.mlp.append(nn.Linear(embed_dim, embed_dim))
            else:
                self.mlp.append(nn.Linear(embed_dim, 1))

    def act(self, graph_feature):
        x_dict, edge_index_dict = graph_feature.x_dict, graph_feature.edge_index_dict

        for i in range(self.num_HGT_layers):
            x_dict = self.conv[i](x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        h_cranes = x_dict["crane"]
        h_machines = x_dict["machine"]
        h_buffers = x_dict["buffer"]
        h_outputs = x_dict["output"]
        h_locations = torch.cat([h_machines, h_buffers, h_outputs], dim=0)
        h_operations = x_dict["operation"]

        h_cranes_pooled = h_cranes.mean(dim=-2)
        h_locations_pooled = h_locations.mean(dim=-2)
        h_operations_pooled = h_operations.mean(dim=-2)

        h_pooled = torch.cat((h_cranes_pooled, h_locations_pooled, h_operations_pooled), dim=-1)

        for i in range(self.num_MLP_layers):
            if i < len(self.mlp) - 1:
                h_pooled = self.mlp[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                state_value = self.mlp[i](h_pooled)

        return state_value.squeeze().item()

    def evaluate(self, batch_graph_feature):
        batch_size = batch_graph_feature.num_graphs
        x_dict, edge_index_dict = batch_graph_feature.x_dict, batch_graph_feature.edge_index_dict

        for i in range(self.num_HGT_layers):
            x_dict = self.conv[i](x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        h_cranes = x_dict["crane"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        h_machines = x_dict["machine"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        h_buffers = x_dict["buffer"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        h_outputs = x_dict["output"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        h_locations = torch.cat([h_machines, h_buffers, h_outputs], dim=1)
        h_operations = x_dict["operation"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)

        h_cranes_pooled = h_cranes.mean(dim=-2)
        h_locations_pooled = h_locations.mean(dim=-2)
        h_operations_pooled = h_operations.mean(dim=-2)

        h_pooled = torch.cat((h_cranes_pooled, h_locations_pooled, h_operations_pooled), dim=-1)

        for i in range(self.num_MLP_layers):
            if i < len(self.mlp) - 1:
                h_pooled = self.mlp[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                batch_state_values = self.mlp[i](h_pooled)

        return batch_state_values