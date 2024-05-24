import numpy as np
import pygame

import bluesky as bs
from envs.common.screen_dummy import ScreenDummy
import envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

AC_DENSITY_RANGE = (0.00000001, 0.0000001) # In AC/NM^2
POLY_AREA_RANGE = (150, 250) # In NM^2
CENTER = (51.990426702297746, 4.376124857109851) # TU Delft AE Faculty coordinates

AC_SPD = 150
AC_TYPE = "A320"

DISTANCE_MARGIN = 5 # km
INTRUSION_DISTANCE = 25 # NM

ACTOR = "KL001"
NM2KM = 1.852

# Model parameters
NUM_AC_STATE = 2
DRIFT_PENALTY = 0
INTRUSION_PENALTY = 0

class PolygonCREnv(gym.Env):
    """ 
    Polygon Conflict Resolution Environment
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}
    
    def __init__(self, render_mode=None):
        self.window_width = 512
        self.window_height = 512
        self.window_size = (self.window_width, self.window_height) # Size of the rendered environment
        self.poly_name = 'airspace'
        self.observation_space = spaces.Dict(
            {
                "cos(drift)": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "sin(drift)": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "x_int": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "y_int": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "vx_int": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64),
                "vy_int": spaces.Box(-np.inf, np.inf, shape=(NUM_AC_STATE,), dtype=np.float64)
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

        self.window = None
        self.clock = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
       
        self._generate_polygon() # Create airspace polygon
        
        self.num_ac = max(np.ceil(np.random.uniform(*AC_DENSITY_RANGE) * self.poly_area), 2) # Get total number of AC in the airspace including agent (min = 2)
        self._generate_ac() # Create aircraft in the airspace
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def _generate_polygon(self):
        
        R = np.sqrt(POLY_AREA_RANGE[1] / np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        self.poly_area = p_area
        
        while p_area < POLY_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        self.poly_points = p # Polygon vertices are saved in terms of NM
        
        p = [fn.nm_to_latlong(CENTER, point) for point in p] # Convert to lat/long coordinateS
        
        points = [coord for point in p for coord in point] # Flatten the list of points
        bs.tools.areafilter.defineArea(self.poly_name, 'POLY', points)
    
    def _generate_waypoints(self):
        
        edges = []
        perim_tot = 0
        
        for i in range(len(self.poly_points)):
            p1 = self.poly_points[i]
            p2 = self.poly_points[(i+1) % len(self.poly_points)] # Ensure wrap-around
            len_edge = fn.euclidean_distance(p1, p2)
            edges.append((p1, p2, len_edge))
            perim_tot += len_edge
        
        d_list = [np.random.uniform(0, perim_tot) for _ in range(self.num_ac)] # Each ac including agent is given a waypoint
        d_list.sort()
        
        self.wpts = []
        current_d = 0
        
        for d in d_list:
            while d > current_d + edges[0][2]:
                current_d += edges[0][2]
                edges.pop(0)
            
            edge = edges[0]
            frac = (d - current_d) / edge[2]
            p = (edge[0][0] + frac * (edge[1][0] - edge[0][0]), edge[0][1] + frac * (edge[1][1] - edge[0][1]))
            self.wpts.append(p)
        
    def _generate_ac(self) -> None:
        
        # Determine bounding box of airspace
        min_x = min(self.poly_points, key=lambda x: x[0])[0]
        min_y = min(self.poly_points, key=lambda x: x[1])[1]
        max_x = max(self.poly_points, key=lambda x: x[0])[0]
        max_y = max(self.poly_points, key=lambda x: x[1])[1]
        
        init_p_latlong = []
        
        while len(init_p_latlong) < self.num_ac:
            p = (np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
            p = fn.nm_to_latlong(CENTER, p)
            
            if bs.tools.areafilter.checkInside(self.poly_name, p[0], p[1], alt=0):
                init_p_latlong.append(p)
        
        wpt_agent = fn.nm_to_latlong(CENTER, self.wpts[0])
        init_pos_agent = init_p_latlong[0]
        hdg_agent = fn.get_hdg(init_pos_agent, wpt_agent)
        
        # Actor AC is the only one that has ACTOR as acid
        bs.traf.cre(ACTOR, actype=AC_TYPE, aclat=init_pos_agent[0], aclon=init_pos_agent[1], achdg=hdg_agent, acspd=AC_SPD)
        
        for i in range(1, len(init_p_latlong)+1):
            wpt = fn.nm_to_latlong(CENTER, self.wpts[i])
            init_pos = init_p_latlong[i]
            hdg = fn.get_hdg(init_pos, wpt)
            bs.traf.cre(acid=i, actype=AC_TYPE, aclat=init_pos[0], aclon=init_pos[1], achdg=hdg, acspd=AC_SPD)
    
    def _get_info(self):
        # Here you implement any additional info that you want to return after a step,
        # but that should not be used by the agent for decision making, so used for logging and debugging purposes
        # for now just have 10, because it crashed if I gave none for some reason.
        return {
            "distance": 10
        }
        
    def _get_obs(self):
        
        ac_idx = bs.traf.id2idx(ACTOR)
        
        # Observation vector shape and components
        self.cos_drift = []
        self.sin_drift = []
        self.x = []
        self.y = []
        self.vx = []
        self.vy = []
        self.x_int = []
        self.y_int = []
        self.vx_int = []
        self.vy_int = []
        
        # Drift of agent aircraft for reward calculation
        drift = 0

        ac_hdg = bs.traf.hdg[ac_idx]
        
        # Get and decompose agent aircaft drift
        wpt_qdr, _  = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpts[0][0], self.wpts[0][1])
        drift = ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)
        self.drift = drift
        self.cos_drift.append(np.cos(np.deg2rad(drift)))
        self.sin_drift.append(np.sin(np.deg2rad(drift)))
        
        # Get agent aircraft position, m
        x, y = fn.latlong_to_nm(CENTER, bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]) * NM2KM * 1000 # Two-step conversion lat/long -> NM -> m
        self.x.append(x)
        self.y.append(y)
        
        # Get agent aircraft velocity, m/s
        # NOTE: TAS or CAS?
        vx = np.cos(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]
        vy = np.sin(np.deg2rad(ac_hdg)) * bs.traf.tas[ac_idx]
        self.vx.append(vx)
        self.vy.append(vy)

        for i in range(self.num_ac-1):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            
            # Intruder AC position, m
            x_int, y_int = fn.latlong_to_nm(CENTER, bs.traf.lat[int_idx], bs.traf.lon[int_idx]) * NM2KM * 1000
            self.x_int.append(x_int)
            self.y_int.append(y_int)
            
            # Intruder AC velocity, m/s
            vx_int = np.cos(np.deg2rad(int_hdg)) * bs.traf.tas[int_idx]
            vy_int = np.sin(np.deg2rad(int_hdg)) * bs.traf.tas[int_idx]
            self.vx_int.append(vx_int)
            self.vy_int.append(vy_int)

        observation = {
            "cos(drift)": np.array(self.cos_drift),
            "sin(drift)": np.array(self.sin_drift),
            "x": np.array(self.x),
            "y": np.array(self.y),
            "vx": np.array(self.vx),
            "vy": np.array(self.vy),
            "x_int": np.array(self.x_int),
            "y_int": np.array(self.y_int),
            "vx_int": np.array(self.vx_int),
            "vy_int": np.array(self.vy_int)
            }
        
        return observation
    
    def _check_drift(self):
        return abs(np.deg2rad(self.drift)) * DRIFT_PENALTY
    
    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx(ACTOR)
        reward = 0
        for i in range(self.num_ac-1):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                reward += INTRUSION_PENALTY
        
        return reward
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        max_distance = 500 # width of screen in km

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))
        
        # Draw airspace
        airspace_color = (255, 0, 0)
        coords = [(point[0]*self.window_width/NM2KM*max_distance, point[1]*self.window_height/NM2KM*max_distance) for point in self.poly_points]
        pygame.draw.polygon(canvas, airspace_color, coords, width = 2)

        # Draw ownship
        ac_idx = bs.traf.id2idx(ACTOR)
        ac_length = 8
        ac_hdg = bs.traf.hdg[ac_idx]
        heading_end_x = ((np.cos(np.deg2rad(ac_hdg)) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(ac_hdg)) * ac_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2-heading_end_x/2,self.window_height/2+heading_end_y/2),
            ((self.window_width/2)+heading_end_x/2,(self.window_height/2)-heading_end_y/2),
            width = 4
        )

        # Draw heading line
        heading_length = 50
        heading_end_x = ((np.cos(np.deg2rad(ac_hdg)) * heading_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(ac_hdg)) * heading_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2,self.window_height/2),
            ((self.window_width/2)+heading_end_x,(self.window_height/2)-heading_end_y),
            width = 1
        )

        # Draw intruders
        ac_length = 3

        for i in range(self.num_ac-1):
            int_idx = i+1
            int_hdg = bs.traf.hdg[int_idx]
            heading_end_x = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width
            heading_end_y = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/max_distance)*self.window_width

            int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

            # Determine color
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
                width = 4
            )

            # Draw heading line
            heading_length = 10
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

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        pass