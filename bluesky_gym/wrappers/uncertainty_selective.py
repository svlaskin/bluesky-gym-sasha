import gymnasium as gym
import numpy as np

class NoisyObservationWrapperSel(gym.Wrapper):
    def __init__(self, env, noise_levels):
        """
        :param env: The environment to wrap.
        :param noise_levels: A list of noise levels corresponding to each observation in the dictionary.
        """
        super().__init__(env)
        self.noise_levels = noise_levels

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        noisy_observation = self.add_noise(observation)
        return noisy_observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        noisy_observation = self.add_noise(observation)
        return noisy_observation, reward, done, truncated, info

    def add_noise(self, observation):
        # Iterate over each observation in the dictionary and apply corresponding noise
        noisy_observation = {}
        for idx, (key, value) in enumerate(observation.items()):
            noise_level = self.noise_levels[idx] if idx < len(self.noise_levels) else self.noise_levels[0]
            noisy_observation[key] = value + np.random.normal(0, noise_level, size=value.shape) \
                if isinstance(value, np.ndarray) else value
        return noisy_observation
