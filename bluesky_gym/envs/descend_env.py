import numpy as np
import pygame

import bluesky as bs
from bluesky.simulation import ScreenIO

import gymnasium as gym
from gymnasium import spaces

class ScreenDummy(ScreenIO):
    """
    Dummy class for the screen. Inherits from ScreenIO to make sure all the
    necessary methods are there. This class is there to reimplement the echo
    method so that console messages are ignored.
    """
    def echo(self, text='', flags=0):
        pass

class DescendEnv(gym.Env):
    """ 
    Very simple environment that requires the agent to climb / descend to a target altitude.
    As the runway approaches the aircraft has to start descending, knowing when to start
    the descend.

    TODO:
    - rendering
    - better commenting
    - proper normalization functionality
    - Monitor Wrapper class for monitoring progress, can be something to be used by all envs.
    """

    # information regarding the possible rendering modes of the environment
    # for BlueSkyGym probably only implement 1 for now together with None, which is default
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 256 # Size of the rendered environment

        self.observation_space = spaces.Dict(
            {
                "altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "vz": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "target_altitude": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "runway_distance": spaces.Box(-np.inf, np.inf, dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')

    def _get_obs(self):
        """
        Observation consists of altitude, vertical speed, target altitude and distance to runway
        Very crude normalization in place for now
        """

        altitude = np.array([(bs.traf.alt[0] - 1500)/3000])
        vz = np.array([bs.traf.vs[0] / 5])
        target_alt = np.array([((self.target_alt- 1500)/3000)])
        runway_distance = np.array([((200 - bs.tools.geo.kwikdist(52,4,bs.traf.lat[0],bs.traf.lon[0])*1.852)-100)/200])

        observation = {
                "altitude": altitude,
                "vz": vz,
                "target_altitude": target_alt,
                "runway_distance": runway_distance,
            }
        
        return observation
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "distance": 10
        }
    def _get_reward(self, observation):

        altitude = (observation['altitude']*3000) + 1500
        target_altitude = (observation['target_altitude']*3000) + 1500
        runway_distance = (observation['runway_distance']*200) + 100

        # reward part of the function
        if runway_distance > 0 and altitude > 0:
            return abs(target_altitude- altitude)*-5/3000, 0
        elif altitude <= 0:
            return -100, 1
        elif runway_distance <= 0:
            return abs(100-altitude)*-50/3000, 1
        
    def _get_action(self,action):
        # Transform action to the meters per second
        action = action * 12.5

        # Bluesky interpretes vertical velocity command through altitude commands 
        # with a vertical speed (magnitude). So check sign of action and give arbitrary 
        # altitude command

        # The actions are then executed through stack commands;
        if action >= 0:
            bs.traf.selalt[0] = 1000000
            bs.traf.selvs[0] = action
        elif action < 0:
            bs.traf.selalt[0] = 0
            bs.traf.selvs[0] = action

    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed)

        alt_init = np.random.randint(2000, 4000)
        self.target_alt = alt_init + np.random.randint(-500,500)

        bs.traf.cre('KL001',actype="A320",acalt=alt_init,acspd=150)
        bs.traf.swvnav[0] = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        
        self._get_action(action)

        action_frequency = 30
        for i in range(action_frequency):
            bs.sim.step()

        observation = self._get_obs()
        reward, terminated = self._get_reward(observation)

        info = self._get_info()

        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info
    
    def render(self):
        pass

    def _render_frame(self):
        pass
        
    def close(self):
        pass