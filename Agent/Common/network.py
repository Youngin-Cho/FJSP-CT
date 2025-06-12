import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv
from torch.distributions import Categorical


class GlobalScheduler(nn.Module):
    def __init__(self,
                 meta_data=None,
                 state_size=None,
                 num_nodes=None,
                 embed_dim=128,
                 num_heads=4,
                 num_HGT_layers=2,
                 num_actor_layers=2,
                 num_critic_layers=2):

        super(GlobalScheduler, self).__init__()

        self.meta_data = meta_data
        self.state_size = state_size
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_actor_layers = num_actor_layers
        self.num_critic_layers = num_critic_layers

        self.num_cranes = self.num_nodes["crane"]
        self.num_locations = self.num_nodes["machine"] + self.num_nodes["buffer"] + self.num_nodes["output"]

        self.conv = nn.ModuleList()
        for i in range(self.num_HGT_layers):
            if i == 0:
                self.conv.append(HGTConv(self.state_size, embed_dim, meta_data, heads=num_heads))
            else:
                self.conv.append(HGTConv(embed_dim, embed_dim, meta_data, heads=num_heads))

        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(8, embed_dim))
        self.fc.append(nn.Linear(embed_dim, embed_dim))

        self.actor = nn.ModuleList()
        for i in range(num_actor_layers):
            if i == 0:
                self.actor.append(nn.Linear(embed_dim * 4, embed_dim))
            elif 0 < i < num_actor_layers - 1:
                self.actor.append(nn.Linear(embed_dim, embed_dim))
            else:
                self.actor.append(nn.Linear(embed_dim, 1))

        self.critic = nn.ModuleList()
        for i in range(num_critic_layers):
            if i == 0:
                self.critic.append(nn.Linear(embed_dim * 3, embed_dim))
            elif i < num_critic_layers - 1:
                self.critic.append(nn.Linear(embed_dim, embed_dim))
            else:
                self.critic.append(nn.Linear(embed_dim, 1))

    def act(self, graph_feature, pairwise_feature, mask, current_operations, reorder_idx, greedy=False):
        x_dict, edge_index_dict = graph_feature.x_dict, graph_feature.edge_index_dict

        for i in range(self.num_HGT_layers):
            x_dict = self.conv[i](x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        h_machines = x_dict["machine"]
        h_buffers = x_dict["buffer"]
        h_outputs = x_dict["output"]
        h_locations = torch.cat([h_machines, h_buffers, h_outputs], dim=0)
        h_locations = torch.index_select(h_locations, 0, reorder_idx)

        h_ops = x_dict["operation"]
        jobs_gather = current_operations.unsqueeze(-1).expand(-1, self.embed_dim)
        h_jobs = h_ops.gather(0, jobs_gather)

        h_cranes = x_dict["crane"]

        h_jobs_padding = h_jobs[:, None, None, :].expand(-1, self.num_locations, self.num_cranes, - 1)
        h_locations_padding = h_locations[None, :, None, :].expand_as(h_jobs_padding)
        h_cranes_padding = h_cranes[None, None, :, :].expand_as(h_jobs_padding)

        h_added = pairwise_feature
        for i in range(self.num_HGT_layers):
            h_added = self.fc[i](h_added)
            h_added = F.elu(h_added)

        h_actions = torch.cat((h_locations_padding, h_jobs_padding, h_cranes_padding, h_added), dim=-1)

        for i in range(self.num_actor_layers):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                logits = self.actor[i](h_actions).flatten()

        mask = mask.transpose(0, 1).flatten()
        logits[~mask] = float('-inf')
        probs = F.softmax(logits, dim=-1)

        dist = Categorical(probs)

        if greedy:
            action = torch.argmax(probs)
            action_logprob = dist.log_prob(action)
        else:
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            while action_logprob < -15:
                action = dist.sample()
                action_logprob = dist.log_prob(action)

        h_locations_pooled = h_locations.mean(dim=-2)
        h_ops_pooled = h_ops.mean(dim=-2)
        h_cranes_pooled = h_cranes.mean(dim=-2)

        h_pooled = torch.cat((h_locations_pooled, h_ops_pooled, h_cranes_pooled), dim=-1)

        for i in range(self.num_critic_layers):
            if i < len(self.critic) - 1:
                h_pooled = self.critic[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                state_value = self.critic[i](h_pooled)

        return action.item(), action_logprob.item(), state_value.squeeze().item()

    def evaluate(self, batch_graph_feature, batch_pairwise_feature, batch_action, batch_mask, batch_current_operations, batch_reorder_idxs):
        batch_size = batch_graph_feature.num_graphs
        x_dict, edge_index_dict = batch_graph_feature.x_dict, batch_graph_feature.edge_index_dict

        for i in range(self.num_HGT_layers):
            x_dict = self.conv[i](x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        h_machines = x_dict["machine"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        h_buffers = x_dict["buffer"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        h_outputs = x_dict["output"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        h_locations = torch.cat([h_machines, h_buffers, h_outputs], dim=1)
        h_locations = torch.index_select(h_locations, 1, batch_reorder_idxs[0])

        h_ops = x_dict["operation"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        jobs_gather = batch_current_operations.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        h_jobs = h_ops.gather(1, jobs_gather)

        h_cranes = x_dict["crane"].reshape(batch_size, -1, self.embed_dim)

        h_jobs_padding = h_jobs[:, :, None, None, :].expand(-1, -1, self.num_locations, self.num_cranes, -1)
        h_locations_padding = h_locations[:, None, :, None, :].expand_as(h_jobs_padding)
        h_cranes_padding = h_cranes[:, None, None, :, :].expand_as(h_jobs_padding)

        h_added = batch_pairwise_feature
        for i in range(self.num_HGT_layers):
            h_added = self.fc[i](h_added)
            h_added = F.elu(h_added)

        h_actions = torch.cat((h_locations_padding, h_jobs_padding, h_cranes_padding, h_added), dim=-1)

        for i in range(self.num_actor_layers):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                batch_logits = self.actor[i](h_actions).flatten(1)

        batch_mask = batch_mask.transpose(1, 2).flatten(1)
        batch_logits[~batch_mask] = float('-inf')
        batch_probs = F.softmax(batch_logits, dim=1)
        batch_dist = Categorical(batch_probs)
        batch_action_logprobs = batch_dist.log_prob(batch_action.squeeze()).unsqueeze(-1)

        h_locations_pooled = h_locations.mean(dim=-2)
        h_ops_pooled = h_ops.mean(dim=-2)
        h_cranes_pooled = h_cranes.mean(dim=-2)

        h_pooled = torch.cat((h_locations_pooled, h_ops_pooled, h_cranes_pooled), dim=-1)

        for i in range(self.num_critic_layers):
            if i < len(self.critic) - 1:
                h_pooled = self.critic[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                batch_state_values = self.critic[i](h_pooled)

        batch_dist_entropys = batch_dist.entropy().unsqueeze(-1)

        return batch_action_logprobs, batch_state_values, batch_dist_entropys


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
        self.global_state_encoding = global_state_encoding

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


class COMACritic(nn.Module):
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
                 output_dim=1200,
                 global_state_encoding="EP"):

        super(COMACritic, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_HGT_layers = num_HGT_layers
        self.num_MLP_layers = num_MLP_layers
        self.global_state_encoding = global_state_encoding

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
                self.mlp.append(nn.Linear(embed_dim, output_dim))

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