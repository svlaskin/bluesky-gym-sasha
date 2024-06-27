from gymnasium.envs.registration import register

def register_envs():
    """Import the envs module so that environments / scenarios register themselves."""
    register(
        id="DescentEnv-v0",
        entry_point="bluesky_gym.envs.descent_env:DescentEnv",
        max_episode_steps=300,
    )

    register(
        id="PlanWaypointEnv-v0",
        entry_point="bluesky_gym.envs.plan_waypoint_env:PlanWaypointEnv",
        max_episode_steps=300,
    )

    register(
        id="HorizontalCREnv-v0",
        entry_point="bluesky_gym.envs.horizontal_cr_env:HorizontalCREnv",
        max_episode_steps=300,
    )

    register(
        id="AMANEnv-v0",
        entry_point="bluesky_gym.envs.aman_env:AmanEnv",
        max_episode_steps=300,
    )

    register(
        id="AMANEnvS-v0",
        entry_point="bluesky_gym.envs.aman_env_single:AmanEnvS",
        max_episode_steps=300,
    )