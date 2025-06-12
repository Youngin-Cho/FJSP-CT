import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch
from Agent.FlexibleJobShop.network import FJSPScheduler
from Agent.CraneTransportation.network import CTScheduler
from Agent.Common.network import GlobalScheduler, GlobalCritic, COMACritic


class RollOutMemory:
    def __init__(self, device):
        self.device = device

        # input variables
        self.fjsp_graph_features = []
        self.fjsp_pairwise_features = []
        self.fjsp_masks = []
        self.ct_graph_features = []
        self.ct_pairwise_features = []
        self.ct_masks = []
        self.global_graph_features = []
        self.global_fjsp_graph_features = []
        self.global_ct_graph_features = []
        self.global_pairwise_features = []
        self.global_masks = []
        self.current_operations = []
        self.reorder_idxs = []

        # other variables
        self.fjsp_actions = []
        self.fjsp_log_probs = []
        self.ct_actions = []
        self.ct_log_probs = []
        self.global_actions = []
        self.global_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        # input variables
        del self.fjsp_graph_features[:]
        del self.fjsp_pairwise_features[:]
        del self.fjsp_masks[:]
        del self.ct_graph_features[:]
        del self.ct_pairwise_features[:]
        del self.ct_masks[:]
        del self.global_graph_features[:]
        del self.global_fjsp_graph_features[:]
        del self.global_ct_graph_features[:]
        del self.global_pairwise_features[:]
        del self.global_masks[:]
        del self.current_operations[:]
        del self.reorder_idxs[:]

        # other variables
        del self.fjsp_actions[:]
        del self.fjsp_log_probs[:]
        del self.ct_actions[:]
        del self.ct_log_probs[:]
        del self.global_actions[:]
        del self.global_log_probs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]

    def put(self,
            fjsp_state=None,
            ct_state=None,
            global_state=None,
            fjsp_action=None,
            fjsp_log_prob=None,
            ct_action=None,
            ct_log_prob=None,
            global_action=None,
            global_log_prob=None,
            reward=None,
            done=None,
            value=None):

        # input variables
        if fjsp_state is not None:
            self.fjsp_graph_features.append(fjsp_state.graph_feature)
            self.fjsp_pairwise_features.append(fjsp_state.pairwise_feature.unsqueeze(0))
            self.fjsp_masks.append(fjsp_state.mask.unsqueeze(0))
            self.current_operations.append(fjsp_state.current_operations.unsqueeze(0))
            self.reorder_idxs.append(fjsp_state.reorder_idx.unsqueeze(0))

        if ct_state is not None:
            self.ct_graph_features.append(ct_state.graph_feature)
            self.ct_pairwise_features.append(ct_state.pairwise_feature.unsqueeze(0))
            self.ct_masks.append(ct_state.mask.unsqueeze(0))

        if global_state is not None:
            if type(global_state) is tuple:
                self.global_fjsp_graph_features.append(global_state[0])
                self.global_ct_graph_features.append(global_state[1])
            else:
                self.global_graph_features.append(global_state.graph_feature)
                if global_state.pairwise_feature is not None:
                    self.global_pairwise_features.append(global_state.pairwise_feature.unsqueeze(0))
                    self.global_masks.append(global_state.mask.unsqueeze(0))
                    self.current_operations.append(global_state.current_operations.unsqueeze(0))
                    self.reorder_idxs.append(global_state.reorder_idx.unsqueeze(0))

        # other variables
        if fjsp_action is not None:
            self.fjsp_actions.append([fjsp_action])
            self.fjsp_log_probs.append([fjsp_log_prob])

        if ct_action is not None:
            self.ct_actions.append([ct_action])
            self.ct_log_probs.append([ct_log_prob])

        if global_action is not None:
            self.global_actions.append([global_action])
            self.global_log_probs.append([global_log_prob])

        self.rewards.append([reward])
        self.dones.append([not done])
        self.values.append([value])

    def get(self, last_value):
        self.values.append([last_value])

        if len(self.fjsp_graph_features) > 0:
            fjsp_graph_features = Batch.from_data_list(self.fjsp_graph_features).to(self.device)
            fjsp_pairwise_features = torch.concat(self.fjsp_pairwise_features).to(self.device)
            fjsp_masks = torch.concat(self.fjsp_masks).to(self.device)
        else:
            fjsp_graph_features = None
            fjsp_pairwise_features = None
            fjsp_masks = None

        if len(self.ct_graph_features) > 0:
            ct_graph_features = Batch.from_data_list(self.ct_graph_features).to(self.device)
            ct_pairwise_features = torch.concat(self.ct_pairwise_features).to(self.device)
            ct_masks = torch.concat(self.ct_masks).to(self.device)
        else:
            ct_graph_features = None
            ct_pairwise_features = None
            ct_masks = None

        if len(self.global_graph_features) > 0:
            global_graph_features = Batch.from_data_list(self.global_graph_features).to(self.device)
        else:
            global_graph_features = None

        if len(self.global_pairwise_features) > 0:
            global_pairwise_features = torch.concat(self.global_pairwise_features).to(self.device)
            global_masks = torch.concat(self.global_masks).to(self.device)
        else:
            global_pairwise_features = None
            global_masks = None

        if len(self.global_fjsp_graph_features) > 0:
            global_fjsp_graph_features = Batch.from_data_list(self.global_fjsp_graph_features).to(self.device)
            global_ct_graph_features = Batch.from_data_list(self.global_ct_graph_features).to(self.device)
        else:
            global_fjsp_graph_features = None
            global_ct_graph_features = None

        if len(self.fjsp_actions) > 0:
            fjsp_actions = torch.from_numpy(np.array(self.fjsp_actions)).type(torch.long).to(self.device)
            fjsp_log_probs = torch.from_numpy(np.array(self.fjsp_log_probs)).type(torch.float32).to(self.device)
        else:
            fjsp_actions = None
            fjsp_log_probs = None

        if len(self.ct_actions) > 0:
            ct_actions = torch.from_numpy(np.array(self.ct_actions)).type(torch.long).to(self.device)
            ct_log_probs = torch.from_numpy(np.array(self.ct_log_probs)).type(torch.float32).to(self.device)
        else:
            ct_actions = None
            ct_log_probs = None

        if len(self.global_actions) > 0:
            global_actions = torch.from_numpy(np.array(self.global_actions)).type(torch.long).to(self.device)
            global_log_probs = torch.from_numpy(np.array(self.global_log_probs)).type(torch.float32).to(self.device)
        else:
            global_actions = None
            global_log_probs = None

        if len(self.current_operations) > 0:
            current_operations = torch.concat(self.current_operations).to(self.device)
            reorder_idxs = torch.concat(self.reorder_idxs).to(self.device)
        else:
            current_operations = None
            reorder_idxs = None

        rewards = torch.from_numpy(np.array(self.rewards)).type(torch.float32).to(self.device)
        dones = torch.from_numpy(np.array(self.dones)).type(torch.float32).to(self.device)
        values = torch.from_numpy(np.array(self.values)).type(torch.float32).to(self.device)

        return (fjsp_graph_features, fjsp_pairwise_features, fjsp_masks, fjsp_actions, fjsp_log_probs,
                ct_graph_features, ct_pairwise_features, ct_masks, ct_actions, ct_log_probs,
                global_graph_features, global_pairwise_features, global_masks, global_actions, global_log_probs,
                global_fjsp_graph_features, global_ct_graph_features,
                rewards, values, dones, current_operations, reorder_idxs)


class Agent:
    def __init__(self,
                 learning_approach="CTDE",
                 global_state_encoding="EP",
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
                 critic_output_dim=1,
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
                 use_coma_advantage=False,
                 device="cpu"):

        self.name = "RL"

        self.learning_approach = learning_approach
        self.global_state_encoding = global_state_encoding
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.P_coeff = P_coeff
        self.V_coeff = V_coeff
        self.E_coeff = E_coeff
        self.use_value_clipping = use_value_clipping
        self.use_coma_advantage = use_coma_advantage
        self.device = device

        self.memory = RollOutMemory(device)
        if learning_approach == "CTCE":
            self.global_network = GlobalScheduler(meta_data=global_meta_data,
                                                  state_size=global_state_size,
                                                  num_nodes=global_num_nodes,
                                                  embed_dim=embed_dim,
                                                  num_heads=num_heads,
                                                  num_HGT_layers=num_HGT_layers,
                                                  num_actor_layers=num_actor_layers,
                                                  num_critic_layers=num_critic_layers).to(device)
            self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr)
            self.global_scheduler = StepLR(optimizer=self.global_optimizer, step_size=lr_step, gamma=lr_decay)
        elif learning_approach == "CTDE":
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

            if self.global_state_encoding == "EP":
                if not use_coma_advantage:
                    self.global_critic = GlobalCritic(global_meta_data=global_meta_data,
                                                      global_state_size=global_state_size,
                                                      global_num_nodes=global_num_nodes,
                                                      embed_dim=embed_dim,
                                                      num_heads=num_heads,
                                                      num_HGT_layers=num_HGT_layers,
                                                      num_MLP_layers=num_critic_layers,
                                                      global_state_encoding=global_state_encoding).to(device)
                else:
                    self.global_critic = COMACritic(global_meta_data=global_meta_data,
                                                    global_state_size=global_state_size,
                                                    global_num_nodes=global_num_nodes,
                                                    embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    num_HGT_layers=num_HGT_layers,
                                                    num_MLP_layers=num_critic_layers,
                                                    output_dim=critic_output_dim,
                                                    global_state_encoding=global_state_encoding).to(device)
            else:
                if not use_coma_advantage:
                    self.global_critic = GlobalCritic(fjsp_meta_data=fjsp_meta_data,
                                                      fjsp_state_size=fjsp_state_size,
                                                      fjsp_num_nodes=fjsp_num_nodes,
                                                      ct_meta_data=ct_meta_data,
                                                      ct_state_size=ct_state_size,
                                                      ct_num_nodes=ct_num_nodes,
                                                      embed_dim=embed_dim,
                                                      num_heads=num_heads,
                                                      num_HGT_layers=num_HGT_layers,
                                                      num_MLP_layers=num_critic_layers,
                                                      global_state_encoding=global_state_encoding).to(device)
                else:
                    self.global_critic = COMACritic(fjsp_meta_data=fjsp_meta_data,
                                                    fjsp_state_size=fjsp_state_size,
                                                    fjsp_num_nodes=fjsp_num_nodes,
                                                    ct_meta_data=ct_meta_data,
                                                    ct_state_size=ct_state_size,
                                                    ct_num_nodes=ct_num_nodes,
                                                    embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    num_HGT_layers=num_HGT_layers,
                                                    num_MLP_layers=num_critic_layers,
                                                    output_dim=critic_output_dim,
                                                    global_state_encoding=global_state_encoding).to(device)
            self.critic_optimizer = optim.Adam(self.global_critic.parameters(), lr=lr)
            self.critic_scheduler = StepLR(optimizer=self.critic_optimizer, step_size=lr_step, gamma=lr_decay)
        elif learning_approach == "IL":
            pass
        else:
            print("Unknown learning approach")

    def put_sample(self,
                   fjsp_state=None,
                   ct_state=None,
                   global_state=None,
                   fjsp_action=None,
                   fjsp_log_prob=None,
                   ct_action=None,
                   ct_log_prob=None,
                   global_action=None,
                   global_log_prob=None,
                   reward=None,
                   done=None,
                   value=None):

        if self.learning_approach == "CTCE":
            self.memory.put(global_state=global_state,
                            global_action=global_action,
                            global_log_prob=global_log_prob,
                            reward=reward,
                            done=done,
                            value=value)
        elif self.learning_approach == "CTDE":
            self.memory.put(fjsp_state=fjsp_state,
                            ct_state=ct_state,
                            global_state=global_state,
                            fjsp_action=fjsp_action,
                            fjsp_log_prob=fjsp_log_prob,
                            ct_action=ct_action,
                            ct_log_prob=ct_log_prob,
                            reward=reward,
                            done=done,
                            value=value)
        else:
            pass

    def get_action(self, local_state=None, global_state=None, scheduling_mode="fjsp"):
        if self.learning_approach == "CTCE":
            self.global_network.eval()
            with torch.no_grad():
                action, log_prob, value = self.global_network.act(graph_feature=global_state.graph_feature,
                                                                  pairwise_feature=global_state.pairwise_feature,
                                                                  mask=global_state.mask,
                                                                  current_operations=global_state.current_operations,
                                                                  reorder_idx=global_state.reorder_idx)

        elif self.learning_approach == "CTDE":
            if scheduling_mode == "fjsp":
                self.fjsp_network.eval()
                with torch.no_grad():
                    action, log_prob = self.fjsp_network.act(graph_feature=local_state.graph_feature,
                                                             pairwise_feature=local_state.pairwise_feature,
                                                             mask=local_state.mask,
                                                             current_operations=local_state.current_operations,
                                                             reorder_idx=local_state.reorder_idx)

                if global_state is not None:
                    self.global_critic.eval()
                    with torch.no_grad():
                        if self.global_state_encoding == "EP":
                            global_graph_feature = Batch.from_data_list([global_state.graph_feature]).to(self.device)
                            value = self.global_critic.evaluate(
                                batch_global_graph_feature=global_graph_feature
                            ).squeeze()
                        else:
                            fjsp_graph_feature = Batch.from_data_list([global_state[0]]).to(self.device)
                            ct_graph_feature = Batch.from_data_list([global_state[1]]).to(self.device)
                            value = self.global_critic.evaluate(
                                batch_fjsp_graph_feature=fjsp_graph_feature,
                                batch_ct_graph_feature=ct_graph_feature,
                            ).squeeze()

                        if self.use_coma_advantage:
                            value = value.cpu().numpy()
                        else:
                            value = value.item()

                else:
                    value = None

            elif scheduling_mode == "ct":
                self.ct_network.eval()
                with torch.no_grad():
                    action, log_prob = self.ct_network.act(graph_feature=local_state.graph_feature,
                                                           pairwise_feature=local_state.pairwise_feature,
                                                           mask=local_state.mask)
                value = None

            else:
                print("Unknown scheduling mode")
        else:
            pass

        return action, log_prob, value

    def train(self, last_value):
        (fjsp_graph_features,
         fjsp_pairwise_features,
         fjsp_masks,
         fjsp_actions,
         fjsp_log_probs,
         ct_graph_features,
         ct_pairwise_features,
         ct_masks,
         ct_actions,
         ct_log_probs,
         global_graph_features,
         global_pairwise_features,
         global_masks,
         global_actions,
         global_log_probs,
         global_fjsp_graph_features,
         global_ct_graph_features,
         rewards,
         values,
         dones,
         current_operations,
         reorder_idxs) = self.memory.get(last_value)

        if not self.use_coma_advantage:
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

        if self.learning_approach == "CTCE":
            self.global_network.train()

            avg_loss = 0.0

            for i in range(self.K_epoch):
                new_log_probs, new_values, dist_entropy \
                    = self.global_network.evaluate(batch_graph_feature=global_graph_features,
                                                 batch_pairwise_feature=global_pairwise_features,
                                                 batch_action=global_actions,
                                                 batch_mask=global_masks,
                                                 batch_current_operations=current_operations,
                                                 batch_reorder_idxs=reorder_idxs)

                ratio = torch.exp(new_log_probs - global_log_probs)

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

                self.global_optimizer.zero_grad()
                loss.mean().backward()
                self.global_optimizer.step()

                avg_loss += loss.mean().item()

            self.memory.clear()

            return avg_loss / self.K_epoch

        elif self.learning_approach == "CTDE":
            self.fjsp_network.train()
            self.ct_network.train()
            self.global_critic.train()

            avg_loss_fjsp = 0.0
            avg_loss_ct = 0.0
            avg_loss_critic = 0.0

            for i in range(self.K_epoch):
                if self.use_coma_advantage:
                    fjsp_new_log_probs, fjsp_dist_entropy, fjsp_policy \
                        = self.fjsp_network.evaluate(batch_graph_feature=fjsp_graph_features,
                                                     batch_pairwise_feature=fjsp_pairwise_features,
                                                     batch_action=fjsp_actions,
                                                     batch_mask=fjsp_masks,
                                                     batch_current_operations=current_operations,
                                                     batch_reorder_idxs=reorder_idxs,
                                                     return_policy=True)

                    ct_new_log_probs, ct_dist_entropy, ct_policy \
                        = self.ct_network.evaluate(batch_graph_feature=ct_graph_features,
                                                   batch_pairwise_feature=ct_pairwise_features,
                                                   batch_action=ct_actions,
                                                   batch_mask=ct_masks,
                                                   return_policy=True)

                    ##COMA advantage##
                    fjsp_policy = fjsp_policy.detach()
                    ct_policy = ct_policy.detach()

                    batch_size, num_fjsp_actions = fjsp_policy.shape
                    _, num_ct_actions = ct_policy.shape

                    values_reshaped = values.squeeze()[:-1].reshape(batch_size, num_fjsp_actions, -1)

                    fjsp_index = ct_actions.unsqueeze(-1).expand(-1, num_fjsp_actions, 1)
                    ct_index = fjsp_actions.unsqueeze(-1).expand(-1, 1, num_ct_actions)

                    fjsp_values = values_reshaped.gather(dim=2, index=fjsp_index).squeeze()
                    fjsp_baseline = (fjsp_values * fjsp_policy).sum().item()
                    ct_values = values_reshaped.gather(dim=1, index=ct_index).squeeze()
                    ct_baselines = (ct_values * ct_policy).sum().item()

                    joint_value = fjsp_values.gather(dim=1, index=fjsp_actions)
                    fjsp_advantage = joint_value - fjsp_baseline
                    ct_advantage = joint_value - ct_baselines

                else:
                    fjsp_new_log_probs, fjsp_dist_entropy \
                        = self.fjsp_network.evaluate(batch_graph_feature=fjsp_graph_features,
                                                     batch_pairwise_feature=fjsp_pairwise_features,
                                                     batch_action=fjsp_actions,
                                                     batch_mask=fjsp_masks,
                                                     batch_current_operations=current_operations,
                                                     batch_reorder_idxs=reorder_idxs)

                    ct_new_log_probs, ct_dist_entropy \
                        = self.ct_network.evaluate(batch_graph_feature=ct_graph_features,
                                                   batch_pairwise_feature=ct_pairwise_features,
                                                   batch_action=ct_actions,
                                                   batch_mask=ct_masks)

                fjsp_ratio = torch.exp(fjsp_new_log_probs - fjsp_log_probs)
                ct_ratio = torch.exp(ct_new_log_probs - ct_log_probs)

                if self.use_coma_advantage:
                    fjsp_surr1 = fjsp_ratio * fjsp_advantage
                    fjsp_surr2 = torch.clamp(fjsp_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * fjsp_advantage
                else:
                    fjsp_surr1 = fjsp_ratio * advantage
                    fjsp_surr2 = torch.clamp(fjsp_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                fjsp_policy_loss = torch.min(fjsp_surr1, fjsp_surr2)

                if self.use_coma_advantage:
                    ct_surr1 = ct_ratio * ct_advantage
                    ct_surr2 = torch.clamp(ct_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * ct_advantage
                else:
                    ct_surr1 = ct_ratio * advantage
                    ct_surr2 = torch.clamp(ct_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                ct_policy_loss = torch.min(ct_surr1, ct_surr2)

                if self.global_state_encoding == "EP":
                    new_values = self.global_critic.evaluate(batch_global_graph_feature=global_graph_features)
                else:
                    new_values = self.global_critic.evaluate(
                        batch_fjsp_graph_feature=global_fjsp_graph_features,
                        batch_ct_graph_feature=global_ct_graph_features)

                if self.use_coma_advantage:
                    expected_values = ((values_reshaped * fjsp_policy.unsqueeze(2) * ct_policy.unsqueeze(1))
                                       .sum(dim=(1, 2), keepdim=True).squeeze(-1))
                    td_target = rewards + self.gamma * expected_values

                    new_values = new_values.reshape(batch_size, num_fjsp_actions, -1)
                    new_values = new_values.gather(dim=2, index=fjsp_index).squeeze()
                    new_values = new_values.gather(dim=1, index=fjsp_actions)

                if self.use_value_clipping:
                    new_values_clipped = values[:-1] + torch.clamp(new_values - values[:-1], -self.eps_clip, self.eps_clip)
                    value_loss_clipped = F.smooth_l1_loss(new_values_clipped, td_target)
                    value_loss_original = F.smooth_l1_loss(new_values, td_target)
                    value_loss = torch.max(value_loss_original, value_loss_clipped)
                else:
                    value_loss = F.smooth_l1_loss(new_values, td_target)

                fjsp_loss = - self.P_coeff * fjsp_policy_loss - self.E_coeff * fjsp_dist_entropy
                ct_loss = - self.P_coeff * ct_policy_loss - self.E_coeff * ct_dist_entropy
                critic_loss = value_loss

                self.fjsp_optimizer.zero_grad()
                self.ct_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                fjsp_loss.mean().backward()
                ct_loss.mean().backward()
                critic_loss.mean().backward()

                self.fjsp_optimizer.step()
                self.ct_optimizer.step()
                self.critic_optimizer.step()

                avg_loss_fjsp += fjsp_loss.mean().item()
                avg_loss_ct += ct_loss.mean().item()
                avg_loss_critic += critic_loss.mean().item()

            self.memory.clear()

            return avg_loss_fjsp / self.K_epoch, avg_loss_ct / self.K_epoch, avg_loss_critic / self.K_epoch

        else:
            pass

    def save_network(self, e, file_dir):
        if self.learning_approach == "CTCE":
            torch.save({"episode": e,
                        "model_state_dict": self.global_network.state_dict(),
                        "optimizer_state_dict": self.global_optimizer.state_dict()},
                       file_dir + "episode-%d.pt" % e)
        elif self.learning_approach == "CTDE":
            torch.save({"episode": e,
                        "model_state_dict": self.fjsp_network.state_dict(),
                        "optimizer_state_dict": self.fjsp_optimizer.state_dict()},
                       file_dir + "FJSP/episode-%d.pt" % e)
            torch.save({"episode": e,
                        "model_state_dict": self.ct_network.state_dict(),
                        "optimizer_state_dict": self.ct_optimizer.state_dict()},
                       file_dir + "CT/episode-%d.pt" % e)
            torch.save({"episode": e,
                        "model_state_dict": self.global_critic.state_dict(),
                        "optimizer_state_dict": self.critic_optimizer.state_dict()},
                       file_dir + "Critic/episode-%d.pt" % e)
        else:
            pass