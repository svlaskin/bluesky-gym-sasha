import time
import numpy as np
import csv
import torch

from bluesky_zoo import merge_v0
from sac_cr_att.actor import MultiHeadAdditiveActorBasic
from sac_cr_att.critic_q import MultiHeadAdditiveCriticQv3Basic
from sac_cr_att.SAC import SAC
from sac_cr_att.replay_buffer import ReplayBuffer

# Load environment with rendering
env = merge_v0.MergeEnv_ATT_sas(render_mode="human")
agents = env.possible_agents
obs_dim = env.observation_space(agents[0]).shape[0]
action_dim = env.action_space(agents[0]).shape[0]
# n_agents = env.num_ac
n_agents = 30

# Logger here
def log_episode(episode, tot_reward, tot_intrusions, tot_drift, filename="log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:  # write header only once
            writer.writerow(["episode", "tot_reward", "tot_intrusions", "tot_drift"])
        writer.writerow([episode, tot_reward, tot_intrusions, tot_drift])

# Build model
actor = MultiHeadAdditiveActorBasic(q_dim=3, kv_dim=7, out_dim=action_dim, num_heads=3)
critic_q = MultiHeadAdditiveCriticQv3Basic(q_dim=5, kv_dim=7, num_heads=3)
critic_q_target = MultiHeadAdditiveCriticQv3Basic(q_dim=5, kv_dim=7, num_heads=3)

# weights_folder = "/Users/sasha/Documents/Code/multiagent_merge/bluesky-gym/sac_cr_att_sas_vlim_20_50k_epi" # for merge
weights_folder = "/Users/sasha/Documents/Code/multiagent_merge/bluesky-gym/sac_cr_att_sas_2pz" # for merge
# weights_folder = "/Users/sasha/Documents/Code/multiagent_merge/bluesky-gym/sac_cr_att_per_clean" # for CR
actor.load_state_dict(torch.load(f"{weights_folder}/actor.pt"))
critic_q.load_state_dict(torch.load(f"{weights_folder}/qf.pt"))
critic_q_target.load_state_dict(torch.load(f"{weights_folder}/qf_target.pt"))

buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents, size=1, batch_size=1)
model = SAC(action_dim=action_dim, buffer=buffer, actor=actor, critic_q=critic_q, critic_q_target=critic_q_target, gamma=0.90)

# Run episodes
for ep in range(5):
    obs, infos = env.reset()
    done = False
    total_rew = 0
    step = 0

    while not done:
        obs_array = np.stack([obs[a] for a in agents])
        with torch.no_grad():
            actions = model.get_action(obs_array)
        action_dict = {a: act for a, act in zip(agents, actions)}
        obs, rews, dones, truncs, infos = env.step(action_dict)
        total_rew += np.mean(list(rews.values()))
        done = list(dones.values())[0] or list(truncs.values())[0]
        # if hasattr(env, "render_mode") and env.render_mode == "human":
        #     env.render()  # Access underlying BlueSky env render directly
        env._render_frame()
        time.sleep(0.05)
        step += 1

    print(f"Episode {ep+1}: Reward = {total_rew:.2f}, Steps = {step}")
