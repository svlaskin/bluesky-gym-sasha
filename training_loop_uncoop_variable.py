"""Reusable SAC training loop for variable-uncooperative SectorCR ATT runs."""

from pathlib import Path
from typing import Optional
import csv

import numpy as np
import torch

from sac_cr_att.actor import MultiHeadAdditiveActorBasic
from sac_cr_att.critic_q import MultiHeadAdditiveCriticQv3Basic
from sac_cr_att.replay_buffer import ReplayBuffer
from sac_cr_att.SAC import SAC


def save_models(model: SAC, weights_folder: str) -> None:
    weights_path = Path(weights_folder)
    weights_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.actor.state_dict(), weights_path / "actor.pt")
    torch.save(model.critic_q.state_dict(), weights_path / "qf.pt")
    torch.save(model.critic_q_target.state_dict(), weights_path / "qf_target.pt")


def run_training_loop(
    env,
    weights_folder: str,
    num_episodes: int,
    train_steps: int,
    max_episode_length: int,
    save_every: int = 10,
    gamma: float = 0.90,
    buffer_size: int = int(4e6),
    batch_size: int = 1024,
    csv_file: Optional[str] = None,
):
    observations, _ = env.reset()
    agents = list(observations.keys())
    if not agents:
        raise RuntimeError("Environment reset returned no agents.")

    first_agent = agents[0]
    action_dim = env.action_space(first_agent).shape[0]
    observation_dim = env.observation_space(first_agent).shape[0]
    n_agents = env.num_ac

    buffer = ReplayBuffer(
        obs_dim=observation_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        size=buffer_size,
        batch_size=batch_size,
    )

    actor = MultiHeadAdditiveActorBasic(q_dim=3, kv_dim=7, out_dim=action_dim, num_heads=3)
    critic_q = MultiHeadAdditiveCriticQv3Basic(q_dim=5, kv_dim=7, num_heads=3)
    critic_q_t = MultiHeadAdditiveCriticQv3Basic(q_dim=5, kv_dim=7, num_heads=3)

    model = SAC(
        action_dim=action_dim,
        buffer=buffer,
        actor=actor,
        critic_q=critic_q,
        critic_q_target=critic_q_t,
        gamma=gamma,
    )

    if csv_file:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "episode_reward",
                    "total_intrusions",
                    "average_drift",
                    "n_uncoop",
                ]
            )

    total_rewards = np.array([])

    for episode in range(num_episodes):
        observations, infos = env.reset()
        agents = list(observations.keys())
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            obs_array = np.array(list(observations.values()))
            act_array = model.get_action(obs_array)
            actions = {agent: action for agent, action in zip(agents, act_array)}

            observations, rewards, dones, truncates, infos = env.step(actions)

            obs_array_n = np.array(list(observations.values()))
            rew_array = np.array(list(rewards.values()))
            episode_reward += float(rew_array.mean())

            if steps < train_steps:
                model.store_transition(obs_array, act_array, obs_array_n, rew_array, False)

            done = bool(list(dones.values())[0] or list(truncates.values())[0])
            if steps > max_episode_length:
                done = True

            steps += 1

        total_rewards = np.append(total_rewards, episode_reward)

        lead_agent = agents[0]
        lead_info = infos.get(lead_agent, {})
        total_intrusions = lead_info.get("total_intrusions", np.nan)
        average_drift = lead_info.get("average_drift", np.nan)
        n_uncoop = len(getattr(env, "agents_uncoop", []))

        if csv_file:
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        episode,
                        episode_reward,
                        total_intrusions,
                        average_drift,
                        n_uncoop,
                    ]
                )

        if episode % save_every == 0:
            avg_rew = total_rewards[-100:].mean()
            print(
                f"episode: {episode}, avg rew: {avg_rew}, "
                f"n_uncoop: {n_uncoop}, total_intrusions: {total_intrusions}"
            )
            save_models(model, weights_folder)

    return model, total_rewards
