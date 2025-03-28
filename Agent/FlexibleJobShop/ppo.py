import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Batch
from Agent.FlexibleJobShop.network import FJSPScheduler


class RollOutMemory:
    def __init__(self, device):
        self.device = device

        # input variables
        self.graph_features = []
        self.pairwise_features = []
        self.masks = []
        self.current_operations = []

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

        # other variables
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]
        del self.log_probs[:]

    def put(self, state, action, reward, done, log_prob, value):
        # input variables
        self.graph_features.append(state.graph_feature)
        self.pairwise_features.append(state.pairwise_feature)
        self.masks.append(state.mask)
        self.current_operations.append(state.current_operations)

        # other variables
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(not done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def get(self, last_value):
        self.values.append(last_value)

        graph_features = Batch.from_data_list(self.graph_features).to(self.device)
        pairwise_features = torch.FloatTensor(self.pairwise_features).to(self.device)
        masks = torch.BoolTensor(self.masks).to(self.device)
        current_operations = torch.LongTensor(self.current_operations).to(self.device)

        actions = torch.LongTensor(self.actions).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        values = torch.FloatTensor(self.values[1:]).to(self.device)
        log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        return (graph_features, pairwise_features, masks, current_operations,
                actions, rewards, values, dones, log_probs)


class FJSPAgent:
    def __init__(self, meta_data,  # 그래프 구조에 대한 정보
                 state_size,  # 노드 타입 별 특성 벡터의 크기
                 num_nodes,  # 노드 타입 별 그래프 내 노드의 개수
                 embed_dim,  # node embedding 크기
                 num_heads,  # HGT layer에서의 attention head의 수
                 num_HGT_layers,  # HGT layer의 개수
                 num_actor_layers,  # actor layer의 개수
                 num_critic_layers,  # critic layer의 개수
                 lr,  # 학습률
                 lr_decay,  # 학습률에 대한 감소비율
                 lr_step,  # 학습률 감소를 위한 스텝 수
                 gamma,  # 감가율
                 lmbda,  # gae 파라미터
                 eps_clip,  # loss function 내 clipping ratio
                 K_epoch,  # 동일 샘플에 대한 update 횟수
                 P_coeff,  # 정책 학습에 대한 가중치
                 V_coeff,  # 가치함수 학습에 대한 가중치
                 E_coeff,  # 엔트로피에 대한 가중치
                 device="cpu"):

        self.name = "RL"

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.P_coeff = P_coeff
        self.V_coeff = V_coeff
        self.E_coeff = E_coeff
        self.device = device

        self.memory = RollOutMemory(device)
        self.network = FJSPScheduler(meta_data, state_size, num_nodes, embed_dim, num_heads,
                                     num_HGT_layers, num_actor_layers, num_critic_layers).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=lr_step, gamma=lr_decay)

    def put_sample(self, state, action, reward, done, log_prob, value):
        self.memory.put(state, action, reward, done, log_prob, value)

    def get_action(self, state):
        self.network.eval()
        with torch.no_grad():
            action, log_prob, value = self.network.act(graph_feature=state.graph_feature,
                                                       pairwise_feature=state.pairwise_feature,
                                                       mask=state.mask,
                                                       current_operations=state.current_operations,
                                                       reorder_idx=state.reorder_idx)
        return action, log_prob, value

    def train(self, last_value):
        self.network.train()

        graph_features, pairwise_features, masks, current_operations, actions, rewards, values, dones, log_probs \
            = self.memory.get(last_value)

        avg_loss = 0.0

        for i in range(self.K_epoch):
            td_target = rewards + self.gamma * values * dones
            delta = td_target - values

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta.flip(dims=(0,)):
                advantage = self.gamma * self.lmbda * advantage + delta_t
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.concat(advantage_lst).unsqueeze(-1).to(self.device)

            new_log_probs, new_values, dist_entropy \
                = self.network.evaluate(batch_graph_feature=graph_features,
                                        batch_pairwise_feature=pairwise_features,
                                        batch_action=actions,
                                        batch_mask=masks,
                                        batch_current_operations=current_operations)

            ratio = torch.exp(new_log_probs - log_probs)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = (- self.P_coeff * torch.min(surr1, surr2)
                    + self.V_coeff * F.smooth_l1_loss(new_values, td_target)
                    - self.E_coeff * dist_entropy)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            avg_loss += loss.mean().item()

        return avg_loss / self.K_epoch

    def save_network(self, e, file_dir):
        torch.save({"episode": e,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)