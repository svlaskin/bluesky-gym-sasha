import time
import numpy as np
import csv
import os
import torch

from bluesky_zoo import merge_v0, sector_cr_v0
from sac_cr_att.actor import MultiHeadAdditiveActorBasic
from sac_cr_att.critic_q import MultiHeadAdditiveCriticQv3Basic
from sac_cr_att.SAC import SAC
from sac_cr_att.replay_buffer import ReplayBuffer

"""
BIG FILE FOR UNCERTAINTY CR EVAL
"""

# Load environment with rendering
need_render = False
env_c = sector_cr_v0.SectorCR_ATT_sas(render_mode='human') if need_render else sector_cr_v0.SectorCR_ATT_sas(render_mode=None)
env_unc = sector_cr_v0.SectorCR_ATT_sas_unc(render_mode='human') if need_render else sector_cr_v0.SectorCR_ATT_sas_unc(render_mode=None)
agents = env_c.possible_agents
obs_dim = env_c.observation_space(agents[0]).shape[0]
action_dim = env_c.action_space(agents[0]).shape[0]
# n_agents = env.num_ac
n_agents = 30
need_render = False

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

# weights_folder = "/Users/sasha/Documents/Code/multiagent_merge/bluesky-gym/sac_cr_att_sas_vlim_20" # for merge
weights_folder_clean = "/Users/sasha/Documents/Code/multiagent_merge/bluesky-gym/sac_unc_cr_att_clean_posonly_20" # trained on ideal
weights_folder_noise = "/Users/sasha/Documents/Code/multiagent_merge/bluesky-gym/sac_unc_cr_att_noise_posonly_20" # trained on noise
# actor.load_state_dict(torch.load(f"{weights_folder}/actor.pt"))
# critic_q.load_state_dict(torch.load(f"{weights_folder}/qf.pt"))
# critic_q_target.load_state_dict(torch.load(f"{weights_folder}/qf_target.pt"))

buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents, size=1, batch_size=1)
model = SAC(action_dim=action_dim, buffer=buffer, actor=actor, critic_q=critic_q, critic_q_target=critic_q_target, gamma=0.90)

# all combinations possible, with structure [folder, env, name]
# run_pars = [[weights_folder_clean, env_c, 'cc'], [weights_folder_clean, env_unc, 'cu'], [weights_folder_noise, env_c, 'uc'], [weights_folder_noise, env_unc, 'uu']]
# run_pars = [[weights_folder_noise, env_unc, 'uu']]

run_pars = [
            [weights_folder_noise, env_c, 'mvpc15'], 
            [weights_folder_noise, env_unc, 'mvpu15']]
run_epis = 5000 # number of episodes to run per combo


for run_par in run_pars:
    # print(run_par)
    log_name = f"logs_unc_cr_3.5std/log_{run_par[2]}.csv"
    weights_folder = run_par[0]
    env = run_par[1]
    name_save = run_par[2]
    actor.load_state_dict(torch.load(f"{weights_folder}/actor.pt"))
    critic_q.load_state_dict(torch.load(f"{weights_folder}/qf.pt"))
    critic_q_target.load_state_dict(torch.load(f"{weights_folder}/qf_target.pt"))
    buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents, size=1, batch_size=1)
    model = SAC(action_dim=action_dim, buffer=buffer, actor=actor, critic_q=critic_q, critic_q_target=critic_q_target, gamma=0.90)
    # empty arrays for logged values
    rewards = []
    intrusions = []
    drift = []

    # Run episodes
    for ep in range(run_epis):
        obs, infos = env.reset()
        done = False
        total_rew = 0
        total_int = 0
        total_drift = 0
        step = 0

        while not done:
            obs_array = np.stack([obs[a] for a in agents])
            with torch.no_grad():
                actions = model.get_action(obs_array)
            action_dict = {a: act for a, act in zip(agents, actions)}
            obs, rews, dones, truncs, infos = env.step(action_dict)
            total_rew += np.mean(list(rews.values()))
            # total_int += infos['DR001']['total_intrusions']
            # total_drift += infos['DR001']['average_drift']
            done = list(dones.values())[0] or list(truncs.values())[0]
            if need_render:
                if hasattr(env, "render_mode") and env.render_mode == "human":
                    env.render()  # Access underlying BlueSky env render directly
                env._render_frame()
                time.sleep(0.05)
            step += 1
            if step >150:
                done=True
        total_int = infos['DR001']['total_intrusions']
        total_drift = infos['DR001']['average_drift']
        total_rew = infos['DR001']['total_reward']
        log_episode(ep, total_rew, total_int, total_drift, filename=log_name)

        print(f"Episode {ep+1}: Reward = {total_rew:.2f}, Steps = {step}")
        print(f"total int is: {infos['DR001']['total_intrusions']}")
