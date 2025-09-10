import csv
import os
from stable_baselines3.common.callbacks import BaseCallback

class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_dir, file_name='training_log.csv', verbose=0):
        super(CSVLoggerCallback, self).__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, file_name)
        self.headers = ['timesteps', 'episodes']
        self.initialized = False
        self.episode_count = 0

    def _on_step(self) -> bool:
        if not self.initialized:
            # Initialize headers based on keys in the infos dictionary
            self.info_keys = self.locals['infos'][0].keys()
            self.headers.extend(self.info_keys)
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
            self.initialized = True

        if self.locals['dones'][0]:
            self.episode_count += 1
            timesteps = self.num_timesteps
            info_dict = self.locals['infos'][0]
            info_values = [info_dict.get(key, None) for key in self.info_keys]
            row = [timesteps, self.episode_count] + list(info_values)
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        return True