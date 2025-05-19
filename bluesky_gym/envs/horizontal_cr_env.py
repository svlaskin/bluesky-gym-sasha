"""
Horizontal CR adapted for M600 performance characteristics, with smaller area and more spawned intruders.
This is used for uncertainty-robust CR training, with togglable noise.
The BASELINE boolean is to activate MVP in the place of the model input, for comparison purposes. This suppresses the action stack commands, and activates RESO MVP.
"""
import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

DISTANCE_MARGIN = 0.5 # km
REACH_REWARD = 1

DRIFT_PENALTY = -0.1/10
INTRUSION_PENALTY = -1/5
NEG_SPEED_PENALTY = -0.1/10

NUM_INTRUDERS = 10
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 0.075 # NM, this is 138m. For testing, 50.
# INTRUSION_DISTANCE = 0.15
# WAYPOINT_DISTANCE_MIN = 10
# WAYPOINT_DISTANCE_MAX = 25
WAYPOINT_DISTANCE_MIN = 5 # 5
WAYPOINT_DISTANCE_MAX = 15 # 10

D_HEADING = 45
D_SPEED = 5

AC_SPD = 10

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 5 # nominal 10

EXTENDED_ACTION_SPACE = True
BASELINE = False

class HorizontalCREnv(gym.Env):
    """ 
    Horizontal Conflict Resolution Environment

    TODO:
    - look at adding waypoints instead of staying straight
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment


        # Observation space should include ownship and intruder info
        # Maybe later also have an option for CNN based intruder info, could be interesting
        self.observation_space = spaces.Dict(
            {
                "intruder_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "x_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "y_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "cos_drift": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                "sin_drift": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64)
            }
        )
       
        if EXTENDED_ACTION_SPACE:
            self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64) # for both
        else:
            self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float64) # for heading only

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF') # nominal is 5

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])
        self.average_v_in = np.array([]) # average speed input
        self.average_h_in = np.array([]) # average heading input
        self.cpa_vec = None #closest point of approach per intruder, initialised as high value
        # self.bearing_cpa_vec = np.array([]) # beraing at CPA,
        # self.pos_cpa_vec =  np.array([]) # ownship position at CPA, unused

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()

        self.total_reward = 0
        self.total_intrusions = 0
        self.average_drift = np.array([])

        bs.traf.cre('DR001',actype="M600",acspd=AC_SPD)
        if BASELINE:
            bs.stack.stack("asas on")
            bs.stack.stack("reso mvp")

        self._generate_conflicts()
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

    def _generate_conflicts(self, acid = 'DR001'):
        target_idx = bs.traf.id2idx(acid)
        for i in range(NUM_INTRUDERS):
            dpsi = np.random.randint(45,315)
            cpa = np.random.uniform(0,INTRUSION_DISTANCE)
            # tlosh = np.random.randint(10,100)
            tlosh = np.random.randint(5,75)
            bs.traf.creconfs(acid=f'{i}',actype="M600",targetidx=target_idx,dpsi=dpsi,dcpa=cpa,tlosh=tlosh)
            if BASELINE:
                bs.stack.stack(f'RESOOFF {i}')
            bs.traf.perf.axmax[-1] = 5 # m/s2, max acceln, overwrite the default.

    def _generate_waypoint(self, acid = 'DR001'):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        for i in range(NUM_WAYPOINTS):
            wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
            wpt_hdg_init = 0

            ac_idx = bs.traf.id2idx(acid)

            wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)    
            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('DR001')

        self.intruder_distance = []
        self.cos_bearing = []
        self.sin_bearing = []
        self.x_difference_speed = []
        self.y_difference_speed = []

        self.waypoint_distance = []
        self.wpt_qdr = []
        self.cos_drift = []
        self.sin_drift = []
        self.drift = []

        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_spd = bs.traf.vs[ac_idx]
        if self.cpa_vec == None:
            self.cpa_vec = self.intruder_distance

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        
            self.intruder_distance.append(int_dis * NM2KM)

            bearing = self.ac_hdg - int_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.sin_bearing.append(np.sin(np.deg2rad(bearing)))

            heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
            x_dif = - np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
            y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

            self.x_difference_speed.append(x_dif)
            self.y_difference_speed.append(y_dif)
        
        # update CPA vector
        mask = self.intruder_distance < self.cpa_vec # if the current is lower than the stored
        self.cpa_vec[mask] = self.intruder_distance[mask]

        for lat, lon in zip(self.wpt_lat, self.wpt_lon):
            
            self.ac_hdg = bs.traf.hdg[ac_idx]
            wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], lat, lon)
        
            self.waypoint_distance.append(wpt_dis * NM2KM)
            self.wpt_qdr.append(wpt_qdr)

            drift = self.ac_hdg - wpt_qdr
            drift = fn.bound_angle_positive_negative_180(drift)

            self.drift.append(drift)
            self.cos_drift.append(np.cos(np.deg2rad(drift)))
            self.sin_drift.append(np.sin(np.deg2rad(drift)))

        observation = {
                "intruder_distance": np.array(self.intruder_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_pos": np.array(self.cos_bearing),
                "sin_difference_pos": np.array(self.sin_bearing),
                "x_difference_speed": np.array(self.x_difference_speed)/AC_SPD,
                "y_difference_speed": np.array(self.y_difference_speed)/AC_SPD,
                "waypoint_distance": np.array(self.waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_drift": np.array(self.cos_drift),
                "sin_drift": np.array(self.sin_drift)
            }
        
        return observation
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            'total_reward': self.total_reward,
            'total_intrusions': self.total_intrusions,
            'average_drift': self.average_drift.mean(),
            'average_v_in': np.mean(self.average_v_in),
            'average_h_in': np.mean(self.average_h_in),
            'cpa': self.cpa_vec
        }

    def _get_reward(self):

        # Always return done as false, as this is a non-ending scenario with 
        # new waypoints spawning continously

        reach_reward = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()
        speed_reward = self._check_speed()

        total_reward = reach_reward + drift_reward + intrusion_reward
        self.total_reward += total_reward

        if 0 in self.wpt_reach:
            return total_reward, 0
        else:
            return total_reward, 1
        
    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.waypoint_distance:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
                index += 1
            else:
                reward += 0
                index += 1
        return reward

    def _check_drift(self):
        drift = abs(np.deg2rad(self.drift[0]))
        self.average_drift = np.append(self.average_drift, drift)
        return drift * DRIFT_PENALTY
    
    def _check_speed(self):
        if self.ac_spd<0:
            return np.abs(NEG_SPEED_PENALTY*self.ac_spd)
        else:
            return 0

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('DR001')
        reward = 0
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                reward += INTRUSION_PENALTY
        
        return reward
    
    def _get_action(self,action):
        dv = action[1] * D_SPEED
        dh = action[0] * D_HEADING
        self.average_h_in = np.append(self.average_h_in, dh)
        self.average_v_in = np.append(self.average_v_in, dv)
        hdg_new = self.ac_hdg + dh
        spd_new = (bs.traf.tas[0] + dv) * MpS2Kt

        if spd_new>0:
            if not BASELINE:
                pass
                # bs.stack.stack(f"SPD DR001 {spd_new}")
        
        # if doing only heading control
        else:
            hdg_new = self.ac_hdg + action * D_HEADING
            if not BASELINE:
                pass
                # bs.stack.stack(f"HDG DR001 {hdg_new}")
        


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 20 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        # draw ownship
        ac_idx = bs.traf.id2idx('DR001')
        ac_length = 0.8
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * ac_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2-heading_end_x/2,self.window_height/2+heading_end_y/2),
            ((self.window_width/2)+heading_end_x/2,(self.window_height/2)-heading_end_y/2),
            width = 1
        )

        # draw heading line
        heading_length = 2
        heading_end_x = ((np.cos(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(bs.traf.hdg[ac_idx])) * heading_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 1
        )

        # draw intruders
        ac_length = 0.3

        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine color
            if int_dis < INTRUSION_DISTANCE:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            x_pos = (self.window_width/2)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (self.window_height/2)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )

            # draw heading line
            heading_length = 2
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/max_distance)*self.window_width

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 1
            )

            pygame.draw.circle(
                canvas, 
                color,
                (x_pos,y_pos),
                radius = (INTRUSION_DISTANCE*NM2KM/max_distance)*self.window_width,
                width = 2
            )

            # import code
            # code.interact(local=locals())

        # draw target waypoint
        for qdr, dis, reach in zip(self.wpt_qdr, self.waypoint_distance, self.wpt_reach):

            circle_x = ((np.cos(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width
            circle_y = ((np.sin(np.deg2rad(qdr)) * dis)/max_distance)*self.window_width

            if reach:
                color = (155,155,155)
            else:
                color = (255,255,255)

            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                ((self.window_width/2)+circle_x,(self.window_height/2)-circle_y),
                radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass