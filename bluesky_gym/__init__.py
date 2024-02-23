from gymnasium.envs.registration import register

def register_envs():
    """Import the envs module so that environments / scenarios register themselves."""
    register(
        id="DescendEnv-v0",
        entry_point="bluesky_gym.envs.descend_env:DescendEnv",
        max_episode_steps=300,
    )