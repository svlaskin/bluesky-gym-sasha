from gymnasium.envs.registration import register

def register_envs():
    """Import the envs module so that environments / scenarios register themselves."""
    register(
        id="DescentEnv-v0",
        entry_point="bluesky_gym.envs.descent_env:DescentEnv",
        max_episode_steps=300,
    )

    register(
        id="WaypointFollowEnv-v0",
        entry_point="bluesky_gym.envs.waypoint_follow_env:WaypointFollowEnv",
        max_episode_steps=300,
    )

    register(
        id="WaypointPlanEnv-v0",
        entry_point="bluesky_gym.envs.waypoint_plan_env:WaypointPlanEnv",
        max_episode_steps=300,
    )