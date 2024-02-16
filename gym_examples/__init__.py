from gymnasium.envs.registration import register

def register_envs():
    register(
        id="GridWorld-v0",
        entry_point="gym_examples.envs.grid_world:GridWorldEnv",
        max_episode_steps=300,
    )