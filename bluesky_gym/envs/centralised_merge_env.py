import numpy as np
import pygame
import sys
# sys.path.insert(0, 'blueskyy/')
import bluesky
import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

import random
import itertools

DISTANCE_MARGIN = 0.5 # km
REACH_REWARD = 1

DRIFT_PENALTY = -0.1
# DRIFT_PENALTY = 0
INTRUSION_PENALTY = -1
VDELT_PENALTY = -0.05

# NUM_WAYPOINTS = 1
# INTRUSION_DISTANCE = 0.4 # NM
INTRUSION_DISTANCE = 0.15 # TODO: M2 numbers
# INTRUSION_DISTANCE = 0.026998

SPAWN_DISTANCE_MIN = 5
SPAWN_DISTANCE_MAX = 20

# INTRUDER_DISTANCE_MIN = 20
# INTRUDER_DISTANCE_MAX = 50

D_HEADING = 45 # 45 degrees each way from desired track
D_SPEED = 0.5
# D_SPEED = 0.1

AC_SPD = 15

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 10

NUM_AC = 5
NUM_AC_STATE = 5
NUM_WAYPOINTS = 1

# Final approach fix, north of EHAM in Heiloo
# FIX_LAT = 52.59382191779792 # change to FAF
# FIX_LON = 4.722605450577005

# end of polderbaan EHAM
RWY_LAT = 52.36239301495972
RWY_LON = 4.713195734579777

# Alternative - FAF at a distance
distance_faf_rwy = 30 # NM, initial assumption 
bearing_faf_rwy = 0
FIX_LAT, FIX_LON = fn.get_point_at_distance(RWY_LAT, RWY_LON, distance_faf_rwy, bearing_faf_rwy)

# spawn location ownship
ACLAT_INIT = 52.97843850741256
ACLON_INIT = 4.511017581418151

class AmanEnvM(gym.Env):
    """ 
    Single-agent arrival manager environment - only one aircraft (ownship) is merged into NPC stream of aircraft.
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 750
        self.window_height = 500
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment


        # Observation space should include info of all aircraft
        # TODO: make this all multidimensional; how is not certain for some of the things.
        # so far, still need: cos sin airspeed waypt_dist faf_reached x_r y_r vx and vy sin and cos track adn distances

        # NOTE: all of these are relative to the FAF!!! 
        self.observation_space = spaces.Dict(
            {
                "cos(drift)": spaces.Box(-1, 1, shape=(NUM_AC_STATE,), dtype=np.float64),
                "sin(drift)": spaces.Box(-1, 1, shape=(NUM_AC_STATE,), dtype=np.float64),
                "airspeed": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "waypoint_dist": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "faf_reached": spaces.Box(0, 1, shape=(NUM_AC_STATE,), dtype=np.float64),
                "x_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "y_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "vx_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "vy_r": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "cos(track)": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "sin(track)": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "distances": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                # "intruder_distance": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE*NUM_AC_STATE,), dtype=np.float64),
                # "intruder_bearing": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE*NUM_AC_STATE,), dtype=np.float64)
                # "cos_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,NUM_AC_STATE), dtype=np.float64),
                # "sin_difference_pos": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,NUM_AC_STATE), dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(2*NUM_AC_STATE,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 5;FF')

        # initialize values used for logging -> input in _get_info
        self.total_reward = 0
        self.average_drift = []
        self.total_intrusions = 0
        self.average_v_action = []
        self.average_hdg_action = []
        self.in_int = np.zeros(NUM_AC) # current intrusion status vector
        self.pair_indices = list(itertools.combinations(np.arange(NUM_AC),2))
        self.faf_reached = np.zeros(NUM_AC)

        self.window = None
        self.clock = None
        self.nac = NUM_AC
        self.wpt_reach = np.zeros(NUM_AC)
        self.ac_idx_by_dist = np.zeros(NUM_AC)
        self.wpt_lat = FIX_LAT
        self.wpt_lon = FIX_LON
        self.rwy_lat = RWY_LAT
        self.rwy_lon = RWY_LON

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.wpt_reach = np.zeros(NUM_AC)
        
        bs.traf.reset()

        self.total_reward = 0
        self.average_drift = []
        self.total_intrusions = 0
        self.faf_reached = np.zeros(NUM_AC)

        self._gen_aircraft()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        
        self._get_action(action)

        for i in range(ACTION_FREQUENCY):
            bs.sim.step()
            if self.render_mode == "human":
                observation = self._get_obs()
                self._render_frame()

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()
        return observation, reward, terminated, False, info
    
    # Generate all aircraft.
    def _gen_aircraft(self):
        for i in range(NUM_AC):
            # randomise position here
            bearing_to_pos = random.uniform(-D_HEADING, D_HEADING) # heading radial towards FAF
            distance_to_pos = random.uniform(SPAWN_DISTANCE_MIN,SPAWN_DISTANCE_MAX) # distance to faf 
            # distance_to_pos = 50
            lat_ac, lon_ac = fn.get_point_at_distance(self.wpt_lat, self.wpt_lon, distance_to_pos, bearing_to_pos)
            # create aircraft
            bs.traf.cre(f'EXP{i}',actype="m600",acspd=AC_SPD,aclat=lat_ac,aclon=lon_ac,achdg=bearing_to_pos-180,acalt=10000)
            bs.stack.stack(f"EXP{i} addwpt {FIX_LAT} {FIX_LON}")
            bs.stack.stack(f"EXP{i} dest {RWY_LAT} {RWY_LON}")
        bs.stack.stack('reso off')
        return

    def _get_obs(self):

        # Observation vector shape and components
        self.cos_drift = np.array([])
        self.sin_drift = np.array([])
        self.airspeed = np.array([])
        self.x_r = np.array([])
        self.y_r = np.array([])
        self.vx_r = np.array([])
        self.vy_r = np.array([])
        self.cos_track = np.array([])
        self.sin_track = np.array([])
        self.distances = np.array([])
        self.intruder_distance = np.array([])
        # self.cos_difference_pos = np.array([])
        # self.sin_difference_pos = np.array([])
        self.intruder_bearing = np.array([])

        # Drift of aircraft for reward calculation
        drift = np.zeros(NUM_AC)

        ac_hdg = bs.traf.hdg
        
        # Get and decompose aircaft drift
        # if self.wpt_reach == 0:
        #     wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat, bs.traf.lon, self.wpt_lat, self.wpt_lon)
        # else:
        #     wpt_qdr, wpt_dist  = bs.tools.geo.kwikqdrdist(bs.traf.lat, bs.traf.lon, self.rwy_lat, self.rwy_lon)
        wpt_qdr, wpt_dist = self._get_drift()

        drift = ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)
        self.drift = drift
        self.cos_drift = np.append(self.cos_drift, np.cos(np.deg2rad(drift)))
        self.sin_drift = np.append(self.sin_drift, np.sin(np.deg2rad(drift)))

        self.waypoint_dist = wpt_dist
        
        # pre-loop: look at nearest aircraft to the FAF
        distances = self.waypoint_dist
        # self.ac_idx_by_dist = np.argsort(distances) # TODO: make use of this properly. for now out
        self.ac_idx_by_dist = np.arange(NUM_AC)

        for i in range(NUM_AC):
            ac_idx = self.ac_idx_by_dist[i] # sorted by distance to the FAF
            hdg = bs.traf.hdg[ac_idx]
            # Get agent aircraft airspeed, m/s
            self.airspeed = np.append(self.airspeed, bs.traf.tas[ac_idx])
            
            # AC relative position to FAF, m
            dist, brg = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, bs.traf.lat[ac_idx],bs.traf.lon[ac_idx]) 
            self.x_r = np.append(self.x_r, (dist * NM2KM * 1000) * np.cos(np.deg2rad(brg)))
            self.y_r = np.append(self.y_r, (dist * NM2KM * 1000) * np.sin(np.deg2rad(brg)))
            
            # AC relative velocity to faf (which is just velocity as it is not moving), m/s
            vx = np.cos(np.deg2rad(hdg)) * bs.traf.tas[ac_idx]
            vy = np.sin(np.deg2rad(hdg)) * bs.traf.tas[ac_idx]
            self.vx_r = np.append(self.vx_r, vx)
            self.vy_r = np.append(self.vy_r, vy)

            # AC relative track, rad
            track = np.arctan2(vy, vx)
            self.cos_track = np.append(self.cos_track, np.cos(track))
            self.sin_track = np.append(self.sin_track, np.sin(track))
            self.distances = np.append(self.distances, distances[ac_idx])

            # now redo all for relative to other AC
            dist_int, brg_int = bs.tools.geo.kwikqdrdist(bs.traf.lat, bs.traf.lon, bs.traf.lat[ac_idx],bs.traf.lon[ac_idx])
            self.intruder_distance = np.append(self.intruder_distance,dist_int)
            self.intruder_bearing = np.append(self.intruder_bearing,brg_int)
            # sin_bearing = np.sin(brg_int)
            # TODO: add the sin and cos bearing since that might help....

        observation = {
            "cos(drift)": np.array(self.cos_drift[:NUM_AC_STATE]),
            "sin(drift)": np.array(self.sin_drift)[:NUM_AC_STATE],
            "airspeed": np.array(self.airspeed)[:NUM_AC_STATE],
            "waypoint_dist": np.array(self.waypoint_dist[:NUM_AC_STATE]/250),
            "faf_reached": np.array(self.wpt_reach),
            "x_r": np.array(self.x_r[:NUM_AC_STATE]/1000000),
            "y_r": np.array(self.y_r[:NUM_AC_STATE]/1000000),
            "vx_r": np.array(self.vx_r[:NUM_AC_STATE]/150),
            "vy_r": np.array(self.vy_r[:NUM_AC_STATE]/150),
            "cos(track)": np.array(self.cos_track[:NUM_AC_STATE]),
            "sin(track)": np.array(self.sin_track[:NUM_AC_STATE]),
            "distances": np.array(self.distances[:NUM_AC_STATE]/250),
            # "intruder_distance": np.array(self.intruder_distance/250),
            # "intruder_bearing": np.array(self.intruder_bearing),
            # "cos_difference_pos": np.array(self.cos_bearing),
            # "sin_difference_pos": np.array(self.sin_bearing)
        }

        return observation
    
    def _get_info(self):
        return {
            "total_reward": self.total_reward,
            "faf_reach": self.faf_reached,
            "average_drift": np.mean(self.average_drift),
            "total_intrusions": self.total_intrusions,
            "average_vinput": np.mean(self.average_v_action),
            "average_hdginput": np.mean(self.average_hdg_action)
        }

    def _get_reward(self):
        reach_reward, done = self._check_waypoint()
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()
        # postfaf_reward = self._check_postfaf() # commented out for now
        # postfaf_reward = 0

        # reward = np.sum(reach_reward)/NUM_AC + np.sum(drift_reward)/NUM_AC + np.sum(intrusion_reward)/NUM_AC + np.sum(postfaf_reward)/NUM_AC
        reward = np.sum(reach_reward) + np.sum(drift_reward) + np.sum(intrusion_reward)

        self.total_reward += reward

        return reward, done

    def _check_waypoint(self):
        reward = np.zeros_like(self.waypoint_dist)
        done = np.zeros_like(self.waypoint_dist)
        
        condition1 = (self.waypoint_dist < DISTANCE_MARGIN) & (self.wpt_reach != 1)
        condition2 = (self.waypoint_dist < 2 * DISTANCE_MARGIN) & (self.wpt_reach == 1)
        
        self.wpt_reach[condition1] = 1
        self.faf_reached[condition1] = 1
        reward[condition1] = REACH_REWARD
        
        self.faf_reached[condition2] = 2
        done[condition2] = 1

        reward = float(np.sum(reward))
        done = 1 if sum(done)==NUM_AC else 0
        
        return np.sum(reward), done

    def _get_drift(self):
        # Vectorize the calculation of wpt_qdr and wpt_dist
        lat = np.array(bs.traf.lat)
        lon = np.array(bs.traf.lon)
        
        # Create masks
        mask_wpt_reach_0 = (self.wpt_reach == 0)
        mask_wpt_reach_1 = (self.wpt_reach != 0)
        
        # Initialize arrays for the results
        wpt_qdr = np.zeros_like(lat)
        wpt_dist = np.zeros_like(lat)
        
        # Calculate for mask_wpt_reach_0
        if np.any(mask_wpt_reach_0):
            wpt_qdr_0, wpt_dist_0 = bs.tools.geo.kwikqdrdist(lat[mask_wpt_reach_0], lon[mask_wpt_reach_0], self.wpt_lat, self.wpt_lon)
            wpt_qdr[mask_wpt_reach_0] = wpt_qdr_0
            wpt_dist[mask_wpt_reach_0] = wpt_dist_0
        
        # Calculate for mask_wpt_reach_1
        if np.any(mask_wpt_reach_1):
            wpt_qdr_1, wpt_dist_1 = bs.tools.geo.kwikqdrdist(lat[mask_wpt_reach_1], lon[mask_wpt_reach_1], self.rwy_lat, self.rwy_lon)
            wpt_qdr[mask_wpt_reach_1] = wpt_qdr_1
            wpt_dist[mask_wpt_reach_1] = wpt_dist_1
        
        return wpt_qdr, wpt_dist
    
    # TODO: vectorise. UNSURE if this works
    def _check_drift(self):
        drift = abs(np.deg2rad(self.drift))
        self.average_drift.append(np.average(drift))
        return drift * DRIFT_PENALTY


    # TODO: vectorise for all AC that are not wpt_reach. THINK THIS NOW WORKS
    def _check_intrusion(self):
        pair_indices = self.pair_indices
        reward = 0
        for pair in pair_indices:
            ind1 = pair[0]
            ind2 = pair[1]
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ind1], bs.traf.lon[ind1], bs.traf.lat[ind2], bs.traf.lon[ind2])
            if int_dis < INTRUSION_DISTANCE:
                self.total_intrusions += 1
                # if not self.wpt_reach:
                reward += INTRUSION_PENALTY
                self.in_int[ind1] = 1
            else:
                # intrusion index for render color
                self.in_int[ind1] = 0
                self.in_int[ind2] = 0
        return reward    
    
    def _check_postfaf(self):
        pair_indices = self.pair_indices
        reward = 0
    #     # for pair in pair_indices:
    #     #     ind1 = pair[0]
    #     #     ind2 = pair[1]
    #     #     # if faf reached, velocity diff should be ZERO
    #     #     if self.faf_reached[ind1]==1 and self.faf_reached[ind2]==1:
    #     #         vdelt_x = abs(self.vx_r[ind1]-self.vx_r[ind2])
    #     #         vdelt_y = abs(self.vy_r[ind1]-self.vy_r[ind2])
    #     #         vdelt = np.sqrt(vdelt_x**2+vdelt_y**2)
    #     #         reward += VDELT_PENALTY*vdelt
    #     #     else:
    #     #         reward = 0

        # rather, try to get the reward in terms of the component in the leader's velocity, namely:
        for pair in pair_indices:
            ind1 = pair[0]
            ind2 = pair[1]
            if self.faf_reached[ind1]==1 and self.faf_reached[ind2]==1:
                vec1 = np.array(self.vx_r[ind1],self.vy_r[ind1])
                vec2 = np.array(self.vx_r[ind2],self.vy_r[ind2])
                vec12 = np.dot(vec1,vec2/np.linalg.norm(vec2))# 1 projected onto 2
                reward += VDELT_PENALTY*(np.sqrt((vec12-vec2)**2)) # projected relative velocity should be zero for ideal merge
        
        return reward    

    # TODO: vectorise for multiple AC
    def _get_action(self,action):
        for i in range(NUM_AC):
            action_index = 2*i # map action to flattened action vector.
            # if not self.wpt_reach:
            dh = action[action_index] * D_HEADING
            # else:
            #     dh = -self.drift
            dv = action[action_index+1] * D_SPEED
            self.average_v_action.append(dv)
            self.average_hdg_action.append(dh)

            heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx(f'EXP{i}')] + dh)
            speed_new = (bs.traf.tas[bs.traf.id2idx(f'EXP{i}')] + dv) * MpS2Kt

            # print(speed_new)
            # if self.faf_reached[i]==0:
            bs.stack.stack(f"HDG EXP{i} {heading_new}")
            bs.stack.stack(f"SPD EXP{i} {speed_new}")
            # else:
            #     bs.stack.stack(f"EXP{i} addwpt {RWY_LAT} {RWY_LON}")


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # max_distance = 500 # width of screen in km
        max_distance = 50

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235)) 

        # circle_x = self.window_width/1.2
        # circle_y = self.window_height/2
        circle_x = self.window_width/3
        circle_y = self.window_height/2

        pygame.draw.circle(
            canvas, 
            (255,255,255),
            (circle_x,circle_y),
            radius = 4,
            width = 0
        )
        
        pygame.draw.circle(
            canvas, 
            (255,255,255),
            (circle_x,circle_y),
            radius = (DISTANCE_MARGIN/max_distance)*self.window_width,
            width = 2
        )
        # draw line to faf
        heading_length = 5000
        heading_end_x = ((np.cos(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
        pygame.draw.line(canvas,
        (0,0,0),
        (circle_x,circle_y),
        (circle_x+heading_end_x/2,circle_y-heading_end_y/2),
        width = 2
        )

        # heading boundary lines
        he_x_l = ((np.cos(np.deg2rad(180+135)) * heading_length)/max_distance)*self.window_width
        he_y_l = ((np.sin(np.deg2rad(180+135)) * heading_length)/max_distance)*self.window_width
        he_x_r = ((np.cos(np.deg2rad(180-135)) * heading_length)/max_distance)*self.window_width
        he_y_r = ((np.sin(np.deg2rad(180-135)) * heading_length)/max_distance)*self.window_width
        pygame.draw.line(canvas,
        (3,252,11),
        (circle_x,circle_y),
        (circle_x+he_x_l/2,circle_y-he_y_l/2),
        width = 4
        )
        pygame.draw.line(canvas,
        (3,252,11),
        (circle_x,circle_y),
        (circle_x+he_x_r/2,circle_y-he_y_r/2),
        width = 4
        )

        # draw rwy start
        rwy_faf_qdr, rwy_faf_dis = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, RWY_LAT, RWY_LON)
        x_pos = (circle_x)+(np.cos(np.deg2rad(rwy_faf_qdr))*(rwy_faf_dis * NM2KM)/max_distance)*self.window_width
        y_pos = (circle_y)-(np.sin(np.deg2rad(rwy_faf_qdr))*(rwy_faf_dis * NM2KM)/max_distance)*self.window_height
        heading_length = 5000
        heading_end_x = ((np.cos(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(180)) * heading_length)/max_distance)*self.window_width
        pygame.draw.line(canvas,
        (255,255,255),
        (x_pos,y_pos),
        (circle_x+heading_end_x/2,circle_y-heading_end_y/2),
        width = 4
        )

        # draw aircraft  
        ac_length = 1

        for i in range(0,NUM_AC):
            int_idx = i
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(self.wpt_lat, self.wpt_lon, bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            # int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[0], bs.traf.lon[0], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # determine color
            # if int_dis < INTRUSION_DISTANCE:
            if self.in_int[i]==1:
                color = (220,20,60)
            else: 
                color = (80,80,80)

            if self.wpt_reach[i]:
                color = (0,255,0)
            # if i==0:
            #     color = (252, 43, 28)

            x_pos = (circle_x)+(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_width
            y_pos = (circle_y)-(np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/max_distance)*self.window_height

            pygame.draw.line(canvas,
                color,
                (x_pos,y_pos),
                ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
                width = 4
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

        # PyGame update
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        pass


# ideas: mask actions after wpt OR leave reward crumb waypoints after faf.
# just blocking actions with if statement does not work as expected...