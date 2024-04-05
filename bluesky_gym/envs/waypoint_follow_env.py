import numpy as np
import pygame

import bluesky as bs
from bluesky.simulation import ScreenIO

import gymnasium as gym
from gymnasium import spaces

DISTANCE_MARGIN = 5 # km
WAYPOINT_DISTANCE_MIN = 50 
WAYPOINT_DISTANCE_MAX = 75

DRIFT_PENALTY = -0.05
REACH_REWARD = 1
AC_SPD = 150

D_HEADING = 45

ACTION_FREQUENCY = 10

class ScreenDummy(ScreenIO):
    """
    Dummy class for the screen. Inherits from ScreenIO to make sure all the
    necessary methods are there. This class is there to reimplement the echo
    method so that console messages are ignored.
    """
    def echo(self, text='', flags=0):
        pass

def bound_angle_positive_negative_180(angle_deg: float) -> float:
    """ maps any angle in degrees to the [-180,180] interval 
    Parameters
    __________
    angle_deg: float
        angle that needs to be mapped (in degrees)
    
    Returns
    __________
    angle_deg: float
        input angle mapped to the interval [-180,180] (in degrees)
    """

    if angle_deg > 180:
        return -(360 - angle_deg)
    elif angle_deg < -180:
        return (360 + angle_deg)
    else:
        return angle_deg

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    a = np.radians(bearing)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d/R) + np.cos(lat1) * np.sin(d/R) * np.cos(a))
    lon2 = lon1 + np.arctan2(
        np.sin(a) * np.sin(d/R) * np.cos(lat1),
        np.cos(d/R) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), np.degrees(lon2)

class WaypointFollowEnv(gym.Env):
    """ 
    Dummy environment for horizontal control and rendering testing.
    Goal of the agent is to contiuously follow the waypoints and cross as many as possible
    to score points, similar to snake, but with euler integration for the turn dynamics.

    For now only heading changes are possible.

    TODO:
    
    """

    # information regarding the possible rendering modes of the environment
    # for BlueSkyGym probably only implement 1 for now together with None, which is default
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment

        self.observation_space = spaces.Dict(
            {
                "waypoint_distance": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "cos_difference": spaces.Box(-np.inf, np.inf, dtype=np.float64),
                "sin_difference": spaces.Box(-np.inf, np.inf, dtype=np.float64)
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

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self):
        """
        Observation consists of distance to the waypoint and heading difference with respect to the waypoint
        in cosine and sine decomposition.

        """

        NM2KM = 1.852

        ac_idx = bs.traf.id2idx('KL001')
        self.ac_hdg = bs.traf.hdg[ac_idx]
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat, self.wpt_lon)
    
        self.wpt_dis = wpt_dis * NM2KM
        self.wpt_qdr = wpt_qdr

        self.drift = self.ac_hdg - wpt_qdr
        self.drift = bound_angle_positive_negative_180(self.drift)

        wpt_cos = np.array([np.cos(np.deg2rad(self.drift))])
        wpt_sin = np.array([np.sin(np.deg2rad(self.drift))])

        observation = {
                "waypoint_distance": np.array([self.wpt_dis/WAYPOINT_DISTANCE_MAX]),
                "cos_difference": wpt_cos,
                "sin_difference": wpt_sin,
            }
        
        return observation
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "distance": 10
        }
    
    def _get_reward(self):

        # Always return done as false, as this is a non-ending scenario with 
        # new waypoints spawning continously

        reach_reward = self._check_waypoint()
        return (abs(self.drift) * DRIFT_PENALTY) + reach_reward, 0

        
    def _get_action(self,action):

        # Transform action to the change in heading
        # action = np.random.randint(-100,100)/100
        action = self.ac_hdg + action * D_HEADING

        bs.stack.stack(f"HDG KL001 {action[0]}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD)

        self._generate_waypoint()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        
        self._get_action(action)

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()

        # bluesky reset?? bs.sim.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info
    
    def render(self):
        pass

    def _generate_waypoint(self, acid = 'KL001'):
        wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
        wpt_hdg_init = np.random.randint(0, 359)

        ac_idx = bs.traf.id2idx(acid)

        wpt_lat, wpt_lon = get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)    
        self.wpt_lat = wpt_lat
        self.wpt_lon = wpt_lon

    def _check_waypoint(self):
        if self.wpt_dis < DISTANCE_MARGIN:
            self._generate_waypoint()
            return REACH_REWARD
        else:
            return 0

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 200 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        # draw ownship
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(self.ac_hdg)) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(self.ac_hdg)) * ac_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 4
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.cos(np.deg2rad(self.ac_hdg)) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(self.ac_hdg)) * heading_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 1
        )

        # draw target waypoint

        circle_x = ((np.cos(np.deg2rad(self.wpt_qdr)) * self.wpt_dis)/max_distance)*self.window_width
        circle_y = ((np.sin(np.deg2rad(self.wpt_qdr)) * self.wpt_dis)/max_distance)*self.window_width

        pygame.draw.circle(
            canvas, 
            (255,255,255),
            ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
            radius = 4,
            width = 0
        )
        
        pygame.draw.circle(
            canvas, 
            (255,255,255),
            ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
            radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
            width = 2
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass