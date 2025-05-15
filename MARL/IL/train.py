import os
import json
import torch
import argparse

from torch.utils.tensorboard import SummaryWriter

from Environment.environment import Factory
from Environment.data import DataGenerator
from Agent.FlexibleJobShop.ppo import FJSPAgent
from Agent.CraneTransportation.ppo import CTAgent
from MARL.IL.validate import evaluate


def get_config():
    parser = argparse.ArgumentParser(description="FJSP-CT")

    parser.add_argument("--no_vessl", action='store_true', help="Disable VESSL")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_record', action='store_true', help="Disable Recording events")

    parser.add_argument("--no_pretraining", action='store_true', help="Disable model loading")
    parser.add_argument("--fjsp_model_path", type=str, default=None, help="fjsp model file path")
    parser.add_argument("--ct_model_path", type=str, default=None, help="ct model file path")

    parser.add_argument("--embed_dim", type=int, default=128, help="node embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="multi-head attention in HGT layers")
    parser.add_argument("--num_HGT_layers", type=int, default=2, help="number of HGT layers")
    parser.add_argument("--num_actor_layers", type=int, default=2, help="number of actor layers")
    parser.add_argument("--num_critic_layers", type=int, default=2, help="number of critic layers")

    parser.add_argument("--num_episodes", type=int, default=2000, help="number of episodes")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="learning rate decay ratio")
    parser.add_argument("--lr_step", type=int, default=100, help="step size to reduce learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.1, help="clipping parameter")
    parser.add_argument("--K_epoch", type=int, default=5, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=20, help="the number of steps to obtain samples")
    parser.add_argument("--P_coeff", type=float, default=1, help="coefficient for policy loss")
    parser.add_argument("--V_coeff", type=float, default=0.5, help="coefficient for value loss")
    parser.add_argument("--E_coeff", type=float, default=0.01, help="coefficient for entropy loss")
    parser.add_argument('--no_value_clipping', action='store_true', help="Disable value clipping")

    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=500, help="Save a model every x episodes")
    parser.add_argument("--reset_every", type=int, default=1, help="Generate new instances every x episodes")

    parser.add_argument("--val_dir", type=str, default=None, help="directory where the validation data are stored")

    return parser.parse_args()


def train(config):
    use_cuda = torch.cuda.is_available() and not config.no_cuda
    use_vessl = False if config.no_vessl else True
    use_saved_model = False if config.no_pretraining else True
    use_recording = False if config.no_record else True

    if use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if use_vessl:
        import vessl
        vessl.init(organization="snu-eng-dgx", project="fjsp-ct", hp=config)

    fjsp_model_path = config.fjsp_model_path
    ct_model_path = config.ct_model_path

    # 인공신경망 관련 파라미터
    embed_dim = config.embed_dim
    num_heads = config.num_heads
    num_HGT_layers = config.num_HGT_layers
    num_actor_layers = config.num_actor_layers
    num_critic_layers = config.num_critic_layers

    # 강화학습 알고리즘 관련 파라미터
    num_episodes = config.num_episodes
    lr = config.lr
    lr_decay = config.lr_decay
    lr_step = config.lr_step
    gamma = config.gamma
    lmbda = config.lmbda
    eps_clip = config.eps_clip
    K_epoch = config.K_epoch
    T_horizon = config.T_horizon
    P_coeff = config.P_coeff
    V_coeff = config.V_coeff
    E_coeff = config.E_coeff
    use_value_clipping = False if config.no_value_clipping else True

    eval_every = config.eval_every
    save_every = config.save_every
    reset_every = config.reset_every

    val_dir = config.val_dir

    with open(val_dir + "setting.json", 'r') as f:
        setting = json.load(f)

    fjsp_model_dir = './output/train/IL/model/%d-%d/%s/' % (setting["num_jobs"], setting["num_machines"], "FJSP")
    ct_model_dir = './output/train/IL/model/%d-%d/%s/' % (setting["num_jobs"], setting["num_machines"], "CT")
    log_dir = './output/train/IL/log/%d-%d/' % (setting["num_jobs"], setting["num_machines"])

    if not os.path.exists(fjsp_model_dir):
        os.makedirs(fjsp_model_dir)

    if not os.path.exists(ct_model_dir):
        os.makedirs(ct_model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_dir + "parameters.json", 'w') as f:
        json.dump(vars(config), f, indent=4)

    data_src = DataGenerator(num_inputs=setting["num_inputs"],
                             num_outputs=setting["num_outputs"],
                             num_machines=setting["num_machines"],
                             num_buffers=setting["num_buffers"],
                             num_jobs=setting["num_jobs"],
                             num_options_min=setting["num_options_min"],
                             num_options_max=setting["num_options_max"],
                             num_operations_min=setting["num_operations_min"],
                             num_operations_max=setting["num_operations_max"],
                             proctime_min=setting["proctime_min"],
                             proctime_max=setting["proctime_max"],
                             inter_arrival_time=setting["inter_arrival_time"],
                             num_cranes=setting["num_cranes"],
                             safety_margin=setting["safety_margin"],
                             x_velocity=setting["x_velocity"],
                             y_velocity=setting["y_velocity"],
                             num_rows=setting["num_rows"],
                             x_spacing=setting["x_spacing"],
                             y_spacing=setting["y_spacing"],
                             division=setting["division"])

    env = Factory(data_src,
                  device=device,
                  algorithm=("RL", "RL"),
                  use_recording=use_recording,
                  return_global_state=False)

    fjsp_agent = FJSPAgent(meta_data=env.fjsp_meta_data,
                           state_size=env.fjsp_state_size,
                           num_nodes=env.fjsp_num_nodes,
                           embed_dim=embed_dim,
                           num_heads=num_heads,
                           num_HGT_layers=num_HGT_layers,
                           num_actor_layers=num_actor_layers,
                           num_critic_layers=num_critic_layers,
                           lr=lr,
                           lr_decay=lr_decay,
                           lr_step=lr_step,
                           gamma=gamma,
                           lmbda=lmbda,
                           eps_clip=eps_clip,
                           K_epoch=K_epoch,
                           P_coeff=P_coeff,
                           V_coeff=V_coeff,
                           E_coeff=E_coeff,
                           use_value_clipping=use_value_clipping,
                           use_local_critic=True,
                           device=device)

    ct_agent = CTAgent(meta_data=env.ct_meta_data,
                       state_size=env.ct_state_size,
                       num_nodes=env.ct_num_nodes,
                       embed_dim=embed_dim,
                       num_heads=num_heads,
                       num_HGT_layers=num_HGT_layers,
                       num_actor_layers=num_actor_layers,
                       num_critic_layers=num_critic_layers,
                       lr=lr,
                       lr_decay=lr_decay,
                       lr_step=lr_step,
                       gamma=gamma,
                       lmbda=lmbda,
                       eps_clip=eps_clip,
                       K_epoch=K_epoch,
                       P_coeff=P_coeff,
                       V_coeff=V_coeff,
                       E_coeff=E_coeff,
                       use_value_clipping=use_value_clipping,
                       use_local_critic=True,
                       device=device)

    if not use_vessl:
        writer = SummaryWriter(log_dir)

    if use_saved_model:
        fjsp_checkpoint = torch.load(fjsp_model_path)
        fjsp_agent.network.load_state_dict(fjsp_checkpoint['model_state_dict'])
        # fjsp_agent.optimizer.load_state_dict(fjsp_checkpoint['optimizer_state_dict'])

        ct_checkpoint = torch.load(ct_model_path)
        ct_agent.network.load_state_dict(ct_checkpoint['model_state_dict'])
        # ct_agent.optimizer.load_state_dict(ct_checkpoint['optimizer_state_dict'])

    with open(log_dir + "fjsp_train_log.csv", 'w') as f:
        f.write('episode, reward, loss, lr\n')

    with open(log_dir + "ct_train_log.csv", 'w') as f:
        f.write('episode, reward, loss, lr\n')

    with open(log_dir + "validation_log.csv", 'w') as f:
        f.write('episode, makespan\n')

    for e in range(1, num_episodes + 1):
        if use_vessl:
            vessl.log(payload={"FJSP_Train/LearnigRate": fjsp_agent.scheduler.get_last_lr()[0]}, step=e)
            vessl.log(payload={"CT_Train/LearnigRate": ct_agent.scheduler.get_last_lr()[0]}, step=e)
        else:
            writer.add_scalar("FJSP_Training/LearningRate", fjsp_agent.scheduler.get_last_lr()[0], e)
            writer.add_scalar("CT_Training/LearningRate", ct_agent.scheduler.get_last_lr()[0], e)

        step = 0
        fjsp_step = 0
        ct_step = 0

        episode_reward = 0.0
        fjsp_episode_average_loss = 0.0
        ct_episode_average_loss = 0.0

        fjsp_state, _ = env.reset()

        while True:
            mode = "fjsp" if env.scheduling_mode == "machine" else "ct"

            if mode == "fjsp":
                fjsp_step += 1

                fjsp_action, fjsp_log_prob, fjsp_value = fjsp_agent.get_action(fjsp_state)

                next_ct_state, _, fjsp_reward, done = env.step(fjsp_action)
                episode_reward += fjsp_reward
            else:
                ct_step += 1

                ct_action, ct_log_prob, ct_value = ct_agent.get_action(ct_state)

                next_fjsp_state, _, ct_reward, done = env.step(ct_action)
                episode_reward += ct_reward

            if mode == "fjsp":
                if ct_step >= 1:
                    ct_agent.put_sample(ct_state, ct_action, ct_reward + fjsp_reward, done, ct_log_prob, ct_value)

                ct_state = next_ct_state
            else:
                if fjsp_step >= 1:
                    fjsp_agent.put_sample(fjsp_state, fjsp_action, fjsp_reward + ct_reward, done, fjsp_log_prob, fjsp_value)

                fjsp_state = next_fjsp_state

            if done or len(fjsp_agent.memory.actions) == T_horizon:
                if done:
                    last_value = 0.0
                else:
                    _, _, last_value = fjsp_agent.get_action(fjsp_state)

                if len(fjsp_agent.memory.actions) > 0:
                    fjsp_episode_average_loss += fjsp_agent.train(last_value)

            if done or len(ct_agent.memory.actions) == T_horizon:
                if done:
                    last_value = 0.0
                else:
                    _, _, last_value = ct_agent.get_action(ct_state)

                if len(ct_agent.memory.actions) > 0:
                    ct_episode_average_loss += ct_agent.train(last_value)

            step += 1

            if done:
                # env.monitor.get_logs("./temp%d.xlsx" % e)
                break

        print("episode: %d | reward: %.4f | fjsp_loss: %.4f | ct_loss: %.4f"
              % (e, episode_reward, fjsp_episode_average_loss / fjsp_step, ct_episode_average_loss / ct_step))

        with open(log_dir + "fjsp_train_log.csv", 'a') as f:
            f.write('%d, %1.4f, %1.4f, %f\n'
                    % (e, episode_reward, fjsp_episode_average_loss / fjsp_step, fjsp_agent.scheduler.get_last_lr()[0]))

        with open(log_dir + "ct_train_log.csv", 'a') as f:
            f.write('%d, %1.4f, %1.4f, %f\n'
                    % (e, episode_reward, ct_episode_average_loss / ct_step, ct_agent.scheduler.get_last_lr()[0]))

        if use_vessl:
            vessl.log(payload={"Train/Reward": episode_reward,
                               "FJSP_Train/Loss": fjsp_episode_average_loss / fjsp_step,
                               "CT_Train/Loss": ct_episode_average_loss / ct_step}, step=e)
        else:
            writer.add_scalar("Common/Reward", episode_reward, e)
            writer.add_scalar("FJSP_Training/Loss", fjsp_episode_average_loss / fjsp_step, e)
            writer.add_scalar("CT_Training/Loss", ct_episode_average_loss / ct_step, e)

        fjsp_agent.scheduler.step()
        ct_agent.scheduler.step()

        if e == 1 or e % eval_every == 0:
            average_makespan = evaluate(fjsp_agent, ct_agent, val_dir)

            with open(log_dir + "validation_log.csv", 'a') as f:
                f.write('%d,%1.4f\n' % (e, average_makespan))

            if use_vessl:
                vessl.log(payload={"Perf/Makespan": average_makespan}, step=e)
            else:
                writer.add_scalar("Common/Makespan", average_makespan, e)

        if e % save_every == 0:
            fjsp_agent.save_network(e, fjsp_model_dir)
            ct_agent.save_network(e, ct_model_dir)

        if e % reset_every == 0:
            env = Factory(data_src,
                          device=device,
                          algorithm=("RL", "RL"),
                          use_recording=use_recording,
                          return_global_state=False)

    if not use_vessl:
        writer.close()


if __name__ == "__main__":
    config = get_config()
    train(config)