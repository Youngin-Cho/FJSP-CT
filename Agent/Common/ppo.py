import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch
from Agent.FlexibleJobShop.network import FJSPScheduler
from Agent.CraneTransportation.network import CTScheduler
from Agent.Common.network import GlobalCritic


class RollOutMemory:
    def __init__(self, device):
        self.device = device

        # input variables
        self.graph_features = []
        self.pairwise_features = []
        self.masks = []
        self.current_operations = []
        self.reorder_idxs = []

        # other variables
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def clear(self):
        # input variables
        del self.graph_features[:]
        del self.pairwise_features[:]
        del self.masks[:]
        del self.current_operations[:]
        del self.reorder_idxs[:]

        # other variables
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]
        del self.log_probs[:]

    def put(self, state, action, reward, done, log_prob, value):
        # input variables
        self.graph_features.append(state.graph_feature)
        self.pairwise_features.append(state.pairwise_feature.unsqueeze(0))
        self.masks.append(state.mask.unsqueeze(0))
        self.current_operations.append(state.current_operations.unsqueeze(0))
        self.reorder_idxs.append(state.reorder_idx.unsqueeze(0))

        # other variables
        self.actions.append([action])
        self.rewards.append([reward])
        self.dones.append([not done])
        self.values.append([value])
        self.log_probs.append([log_prob])

    def get(self, last_value):
        self.values.append([last_value])

        graph_features = Batch.from_data_list(self.graph_features).to(self.device)
        pairwise_features = torch.concat(self.pairwise_features).to(self.device)
        masks = torch.concat(self.masks).to(self.device)
        current_operations = torch.concat(self.current_operations).to(self.device)
        reorder_idxs = torch.concat(self.reorder_idxs).to(self.device)

        actions = torch.from_numpy(np.array(self.actions)).type(torch.long).to(self.device)
        rewards = torch.from_numpy(np.array(self.rewards)).type(torch.float32).to(self.device)
        dones = torch.from_numpy(np.array(self.dones)).type(torch.float32).to(self.device)
        values = torch.from_numpy(np.array(self.values)).type(torch.float32).to(self.device)
        log_probs = torch.from_numpy(np.array(self.log_probs)).type(torch.float32).to(self.device)

        return (graph_features, pairwise_features, masks, current_operations, reorder_idxs,
                actions, rewards, values, dones, log_probs)


class Agent:
    def __init__(self,
                 learning_approach="CTDE",
                 fjsp_meta_data=None,  # 그래프 구조에 대한 정보
                 fjsp_state_size=None,  # 노드 타입 별 특성 벡터의 크기
                 fjsp_num_nodes=None,  # 노드 타입 별 그래프 내 노드의 개수
                 ct_meta_data=None,  # 그래프 구조에 대한 정보
                 ct_state_size=None,  # 노드 타입 별 특성 벡터의 크기
                 ct_num_nodes=None,  # 노드 타입 별 그래프 내 노드의 개수
                 global_meta_data=None,  # 그래프 구조에 대한 정보
                 global_state_size=None,  # 노드 타입 별 특성 벡터의 크기
                 global_num_nodes=None,  # 노드 타입 별 그래프 내 노드의 개수
                 embed_dim=128,  # node embedding 크기
                 num_heads=4,  # HGT layer에서의 attention head의 수
                 num_HGT_layers=2,  # HGT layer의 개수
                 num_actor_layers=2,  # actor layer의 개수
                 num_critic_layers=2,  # critic layer의 개수
                 lr=0.0001,  # 학습률
                 lr_decay=1.0,  # 학습률에 대한 감소비율
                 lr_step=100,  # 학습률 감소를 위한 스텝 수
                 gamma=0.98,  # 감가율
                 lmbda=0.95,  # gae 파라미터
                 eps_clip=0.1,  # loss function 내 clipping ratio
                 K_epoch=5,  # 동일 샘플에 대한 update 횟수
                 P_coeff=1.0,  # 정책 학습에 대한 가중치
                 V_coeff=0.5,  # 가치함수 학습에 대한 가중치
                 E_coeff=0.1,  # 엔트로피에 대한 가중치
                 use_value_clipping=True,
                 device="cpu"):

        self.name = "RL"

        self.learning_approach = learning_approach
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.P_coeff = P_coeff
        self.V_coeff = V_coeff
        self.E_coeff = E_coeff
        self.use_value_clipping = use_value_clipping
        self.device = device

        if learning_approach == "CL":
            pass
        elif learning_approach == "CTDE":
            self.fjsp_memory = RollOutMemory(device)
            self.fjsp_network = FJSPScheduler(meta_data=fjsp_meta_data,
                                              state_size=fjsp_state_size,
                                              num_nodes=fjsp_num_nodes,
                                              embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              num_HGT_layers=num_HGT_layers,
                                              num_actor_layers=num_actor_layers,
                                              num_critic_layers=num_critic_layers,
                                              use_local_critic=False).to(device)
            self.fjsp_optimizer = optim.Adam(self.fjsp_network.parameters(), lr=lr)
            self.fjsp_scheduler = StepLR(optimizer=self.fjsp_optimizer, step_size=lr_step, gamma=lr_decay)

            self.ct_memory = RollOutMemory(device)
            self.ct_network = CTScheduler(meta_data=ct_meta_data,
                                          state_size=ct_state_size,
                                          num_nodes=ct_num_nodes,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          num_HGT_layers=num_HGT_layers,
                                          num_actor_layers=num_actor_layers,
                                          num_critic_layers=num_critic_layers,
                                          use_local_critic=False).to(device)
            self.ct_optimizer = optim.Adam(self.ct_network.parameters(), lr=lr)
            self.ct_scheduler = StepLR(optimizer=self.ct_optimizer, step_size=lr_step, gamma=lr_decay)

            self.global_critic = GlobalCritic(meta_data=global_meta_data,
                                              state_size=global_state_size,
                                              num_nodes=global_num_nodes,
                                              embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              num_HGT_layers=num_HGT_layers,
                                              num_MLP_layers=num_critic_layers)
            self.critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=lr)
            self.critic_scheduler = StepLR(optimizer=self.critic_optimizer, step_size=lr_step, gamma=lr_decay)
        elif learning_approach == "IL":
            pass
        else:
            print("Unknown learning approach")

    def put_sample(self, state, action, reward, done, log_prob, value, scheduling_mode):
        if self.learning_approach == "CL":
            pass
        elif self.learning_approach == "CTDE":
            if scheduling_mode == "fjsp":
                self.fjsp_memory.put(state, action, reward, done, log_prob, value)
            elif scheduling_mode == "ct":
                self.ct_memory.put(state, action, reward, done, log_prob, value)
            else:
                print("Unknown scheduling mode")
        else:
            pass

    def get_action(self, state, scheduling_mode):
        if self.learning_approach == "CL":
            pass
        elif self.learning_approach == "CTDE":
            if scheduling_mode == "fjsp":
                self.fjsp_network.eval()
                with torch.no_grad():
                    action, log_prob, value = self.fjsp_network.act(graph_feature=state.graph_feature,
                                                                    pairwise_feature=state.pairwise_feature,
                                                                    mask=state.mask,
                                                                    current_operations=state.current_operations,
                                                                    reorder_idx=state.reorder_idx)
            elif scheduling_mode == "ct":
                self.ct_network.eval()
                with torch.no_grad():
                    action, log_prob, value = self.ct_network.act(graph_feature=state.graph_feature,
                                                                  pairwise_feature=state.pairwise_feature,
                                                                  mask=state.mask)
            else:
                print("Unknown scheduling mode")
        else:
            pass

        return action, log_prob, value

    def train(self, last_value):
        if self.learning_approach == "CL":
            pass
        elif self.learning_approach == "CTDE":
            self.fjsp_network.train()
            self.ct_network.train()

            (fjsp_graph_features, fjsp_pairwise_features, fjsp_masks, current_operations, reorder_idxs,
             fjsp_actions, fjsp_rewards, values, dones, fjsp_log_probs) \
                = self.fjsp_memory.get(last_value)

            (ct_graph_features, ct_pairwise_features, masks,
             ct_actions, rewards, values, dones, ct_log_probs) \
                = self.ct_memory.get(last_value)

        avg_loss = 0.0

        for i in range(self.K_epoch):
            td_target = rewards + self.gamma * values[1:] * dones
            delta = td_target - values[:-1]

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta.flip(dims=(0,)):
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.concat(advantage_lst).unsqueeze(-1).to(self.device)

            # advantage = ((advantage - advantage.mean(dim=1, keepdim=True))
            #               / (advantage.std(dim=1, correction=0, keepdim=True) + 1e-8))

            new_log_probs, new_values, dist_entropy \
                = self.network.evaluate(batch_graph_feature=graph_features,
                                        batch_pairwise_feature=pairwise_features,
                                        batch_action=actions,
                                        batch_mask=masks,
                                        batch_current_operations=current_operations,
                                        batch_reorder_idxs=reorder_idxs)

            ratio = torch.exp(new_log_probs - log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            policy_loss = torch.min(surr1, surr2)

            if self.use_value_clipping:
                new_values_clipped = values[:-1] + torch.clamp(new_values - values[:-1], -self.eps_clip, self.eps_clip)
                value_loss_clipped = F.smooth_l1_loss(new_values_clipped, td_target)
                value_loss_original = F.smooth_l1_loss(new_values, td_target)
                value_loss = torch.max(value_loss_original, value_loss_clipped)
            else:
                value_loss = F.smooth_l1_loss(new_values, td_target)

            loss = - self.P_coeff * policy_loss + self.V_coeff * value_loss - self.E_coeff * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            avg_loss += loss.mean().item()

        self.memory.clear()

        return avg_loss / self.K_epoch

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode-%d.pt" % e)