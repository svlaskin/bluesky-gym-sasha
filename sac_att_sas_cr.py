from bluesky_zoo import sector_cr_v0, merge_v0

from sac_cr_att.actor import MultiHeadAdditiveActorBasic
from sac_cr_att.critic_q  import MultiHeadAdditiveCriticQv3Basic
from sac_cr_att.replay_buffer import ReplayBuffer
from sac_cr_att.SAC import SAC

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import numpy as np
import torch

"""
unc cr runs here, for ideal case
"""

def plot_figures(self, model):
        fig, ax = plt.subplots()
        ax.plot(model.qf1_lossarr, label='qf1')
        ax.plot(model.qf2_lossarr, label='qf2')
        ax.set_yscale('log')
        fig.savefig(self.output_folder+'/qloss.png')
        plt.close(fig)

# def save_models(model, weights_folder = 'sac_cr_att/weights'):
def save_models(model, weights_folder = '/Users/sasha/Documents/Code/multiagent_merge/bluesky-gym/sac_unc_cr_att_clean_posonly_20_3.5n_30rpz'):
    torch.save(model.actor.state_dict(), weights_folder+"/actor.pt")
    torch.save(model.critic_q.state_dict(), weights_folder+"/qf.pt")
    torch.save(model.critic_q_target.state_dict(), weights_folder+"/qf_target.pt")


env = sector_cr_v0.SectorCR_ATT_sas(render_mode=None)
# env = merge_v0.MergeEnv_ATT_sas(render_mode=None)

action_dim = env.action_space('DR001').shape[0] 
observation_dim = env.observation_space('DR001').shape[0]
n_agents = env.num_ac 

num_episodes = 5_000 # 100_000 
train_steps = 500 # first n transitions used for training, to control complexity of samples
max_episode_length = 150

Buffer = ReplayBuffer(obs_dim = observation_dim,
                      action_dim = action_dim,
                      n_agents = n_agents,
                      size = int(4e6),
                      batch_size = 1024)

Actor = MultiHeadAdditiveActorBasic(q_dim = 3,
                                    kv_dim = 7,
                                    out_dim = action_dim,
                                    num_heads = 3)

Critic_q = MultiHeadAdditiveCriticQv3Basic(q_dim = 5,
                                        kv_dim = 7,
                                        num_heads = 3)

Critic_q_t = MultiHeadAdditiveCriticQv3Basic(q_dim = 5,
                                        kv_dim = 7,
                                        num_heads = 3)
model = SAC(action_dim=action_dim,
            buffer = Buffer,
            actor = Actor,
            critic_q = Critic_q,
            critic_q_target= Critic_q_t,
            gamma = 0.90)

observations, infos = env.reset()

agents = list(observations.keys())

obs_array = np.array(list(observations.values()))
act_array = model.get_action(obs_array)

actions = {agent: action for agent, action in zip(agents,act_array)}

observations, rewards, dones, truncates, infos = env.step(actions)

obs_array_n = np.array(list(observations.values()))
rew_array = np.array(list(rewards.values()))
done = list(dones.values())[0]

# model.store_transition(obs_array,act_array,obs_array_n,rew_array,done)

total_rew = np.array([])

for episode in range(num_episodes):
    observations, infos = env.reset()
    done = False
    rew = 0
    steps = 0
    while not done:
        obs_array = np.array(list(observations.values()))
        act_array = model.get_action(obs_array)

        actions = {agent: action for agent, action in zip(agents,act_array)}

        observations, rewards, dones, truncates, infos = env.step(actions)

        obs_array_n = np.array(list(observations.values()))
        rew_array = np.array(list(rewards.values()))
        rew += rew_array.mean()

        if list(dones.values())[0] or list(truncates.values())[0]:
            done = True

        if steps < train_steps:
            model.store_transition(obs_array,act_array,obs_array_n,rew_array,False)
        
        if steps > max_episode_length:
             done = True

        steps += 1

    total_rew = np.append(total_rew,rew)
    if episode % 10 == 0:
        print(f'episode: {episode}, avg rew: {total_rew[-100:].mean()}')
        save_models(model)


import code
code.interact(local=locals())

# create the numpy arrays:
# obs_array = np.array(list(observations.values()))
# would be nice to have this in a wrapper and just create an array environment
