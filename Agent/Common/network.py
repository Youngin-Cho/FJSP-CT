import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv


class GlobalScheduler(nn.Module):
    pass


class GlobalCritic(nn.Module):
    def __init__(self,
                 fjsp_meta_data=None,
                 fjsp_state_size=None,
                 fjsp_num_nodes=None,
                 ct_meta_data=None,
                 ct_state_size=None,
                 ct_num_nodes=None,
                 global_meta_data=None,
                 global_state_size=None,
                 global_num_nodes=None,
                 embed_dim=128,
                 num_heads=4,
                 num_HGT_layers=2,
                 num_MLP_layers=2,
                 global_state_encoding="EP"):

        super(GlobalCritic, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_MLP_layers = num_MLP_layers
        self.global_state = global_state_encoding

        if global_state_encoding == "EP":
            self.global_meta_data = global_meta_data
            self.global_state_size = global_state_size
            self.global_num_nodes = global_num_nodes

            self.conv = nn.ModuleList()
            for i in range(self.num_HGT_layers):
                if i == 0:
                    self.conv.append(HGTConv(global_state_size, embed_dim, global_meta_data, heads=num_heads))
                else:
                    self.conv.append(HGTConv(embed_dim, embed_dim, global_meta_data, heads=num_heads))

        elif global_state_encoding == "CL":
            self.fjsp_state_size = fjsp_state_size
            self.fjsp_meta_data = fjsp_meta_data
            self.fjsp_num_nodes = fjsp_num_nodes

            self.ct_state_size = ct_state_size
            self.ct_meta_data = ct_meta_data
            self.ct_num_nodes = ct_num_nodes

            self.fjsp_conv = nn.ModuleList()
            self.ct_conv = nn.ModuleList()
            for i in range(self.num_HGT_layers):
                if i == 0:
                    self.fjsp_conv.append(HGTConv(fjsp_state_size, embed_dim, fjsp_meta_data, heads=num_heads))
                    self.ct_conv.append(HGTConv(ct_state_size, embed_dim, ct_meta_data, heads=num_heads))
                else:
                    self.fjsp_conv.append(HGTConv(embed_dim, embed_dim, fjsp_meta_data, heads=num_heads))
                    self.ct_conv.append(HGTConv(embed_dim, embed_dim, ct_meta_data, heads=num_heads))

        else:
            print("Invalid global state")

        self.mlp = nn.ModuleList()
        for i in range(num_MLP_layers):
            if i == 0:
                self.mlp.append(nn.Linear(embed_dim * 3, embed_dim))
            elif i < num_MLP_layers - 1:
                self.mlp.append(nn.Linear(embed_dim, embed_dim))
            else:
                self.mlp.append(nn.Linear(embed_dim, 1))

    def evaluate(self,
                 batch_global_graph_feature=None,
                 batch_fjsp_graph_feature=None,
                 batch_ct_graph_feature=None):

        if self.global_state_encoding == "EP":
            assert batch_global_graph_feature is not None

            batch_size = batch_global_graph_feature.num_graphs
            global_x_dict, global_edge_index_dict \
                = batch_global_graph_feature.x_dict, batch_global_graph_feature.edge_index_dict

            for i in range(self.num_HGT_layers):
                global_x_dict = self.conv[i](global_x_dict, global_edge_index_dict)
                global_x_dict = {key: F.elu(x) for key, x in global_x_dict.items()}

            h_cranes = global_x_dict["crane"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_machines = global_x_dict["machine"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_buffers = global_x_dict["buffer"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_outputs = global_x_dict["output"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_locations = torch.cat([h_machines, h_buffers, h_outputs], dim=1)
            h_operations = global_x_dict["operation"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)

        elif self.global_state_encoding == "CL":
            assert batch_fjsp_graph_feature is not None
            assert batch_ct_graph_feature is not None

            batch_size = batch_fjsp_graph_feature.num_graphs
            fjsp_x_dict, fjsp_edge_index_dict \
                = batch_fjsp_graph_feature.x_dict, batch_fjsp_graph_feature.edge_index_dict
            ct_x_dict, ct_edge_index_dict \
                = batch_ct_graph_feature.x_dict, batch_ct_graph_feature.edge_index_dict

            for i in range(self.num_HGT_layers):
                fjsp_x_dict = self.fjsp_conv[i](fjsp_x_dict, fjsp_edge_index_dict)
                fjsp_x_dict = {key: F.elu(x) for key, x in fjsp_x_dict.items()}

                ct_x_dict = self.ct_conv[i](ct_x_dict, ct_edge_index_dict)
                ct_x_dict = {key: F.elu(x) for key, x in ct_x_dict.items()}

            h_cranes = ct_x_dict["crane"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_machines = fjsp_x_dict["machine"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_buffers = fjsp_x_dict["buffer"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_outputs = fjsp_x_dict["output"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
            h_locations = torch.cat([h_machines, h_buffers, h_outputs], dim=1)
            h_operations = (fjsp_x_dict["operation"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
                            + ct_x_dict["operation"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)) * 0.5

        else:
            print("Invalid global state")

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