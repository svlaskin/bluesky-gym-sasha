import gymnasium as gym
import numpy as np


'''
Version of the Noise Wrapper with selective indexing of observation inputs
'''

class NoisyObservationWrapperSel(gym.Wrapper):
    def __init__(self, env, noise_level=0.1, select_indices=[]):
        super().__init__(env)
        self.noise_level = noise_level
        self.select_indices = select_indices

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        noisy_observation = self.add_noise(observation)
        return noisy_observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        noisy_observation = self.add_noise(observation)
        return noisy_observation, reward, done, truncated, info

def add_noise(self, observation):
    # Ensure selected indices are a NumPy array
    selected_indices = np.asarray(self.selected_indices, dtype=int)

    # Validate that selected indices are within bounds
    if len(selected_indices) == 0:
        raise ValueError("No indices selected for noise application.")
    
    if np.any((selected_indices < 0) | (selected_indices >= len(observation))):
        raise IndexError("One or more selected indices are out of bounds.")

    # Handle noise_level cases
    if isinstance(self.noise_level, (int, float)):  
        noise_level = np.full(len(selected_indices), self.noise_level, dtype=np.float64)
    elif isinstance(self.noise_level, (list, tuple, np.ndarray)):
        noise_level = np.asarray(self.noise_level, dtype=np.float64)
        if noise_level.shape != selected_indices.shape:
            raise ValueError("Mismatch: noise_level must be a single value or a list/array of the same length as selected_indices.")
    else:
        raise TypeError("noise_level must be a float, int, list, tuple, or NumPy array.")

    # Generate and apply Gaussian noise at selected indices
    noise = np.random.normal(0, noise_level, size=len(selected_indices))
    noisy_observation = observation.copy()
    noisy_observation[selected_indices] += noise

    return noisy_observation