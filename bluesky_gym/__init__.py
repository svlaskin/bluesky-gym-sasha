from gymnasium.envs.registration import register

def register_envs():
    """Import the envs module so that environments / scenarios register themselves."""
    
    register(
        id="PolygonCREnv-v0",
        entry_point="bluesky_gym.envs.polygon_conf_env:PolygonCREnv",
        max_episode_steps=100,
    )