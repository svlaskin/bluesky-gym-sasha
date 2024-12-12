import gymnasium as gym
from gymnasium.spaces import Dict, Box
import bluesky as bs
import numpy as np

class WindFieldWrapper(gym.Wrapper):
    def __init__(self, env, lat, lon, vnorth, veast, alt, augment_obs=False):
        super().__init__(env)
        self.lat = lat
        self.lon = lon
        self.vnorth = vnorth
        self.veast = veast
        self.alt = alt
        self.augment_obs = augment_obs

        if self.augment_obs:
             #Ensure the original observation space is a Dict
            assert isinstance(self.observation_space, Dict), "This wrapper only supports Dict observation spaces."

            # Add two new components (e.g., 'wind_u' and 'wind_v') to the observation space
            self.observation_space = Dict({
                **self.observation_space.spaces,  # Keep the original observation space entries
                "wind_u": Box(-np.inf, np.inf, shape=(), dtype=np.float64),
                "wind_v": Box(-np.inf, np.inf, shape=(), dtype=np.float64),
            })

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        bs.traf.wind.addpointvne(self.lat, self.lon, self.vnorth, self.veast, self.alt)

        if self.augment_obs: 
            wind_u, wind_v = self._get_wind_observation()
            wind = {
                "wind_u": wind_u,
                "wind_v": wind_v
            }
            observation = {**observation, **wind} 

        return observation, info
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)

        if self.augment_obs: 
            wind_u, wind_v = self._get_wind_observation()
            wind = {
                "wind_u": wind_u,
                "wind_v": wind_v
            }
            observation = {**observation, **wind}

        return observation, reward, done, truncated, info
    
    def _get_wind_observation(self):
        pass