import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1 # reach set waypoint

DRIFT_PENALTY = -0.1
AC_INTRUSION_PENALTY = -1 
RESTRICTED_AREA_INTRUSION_PENALTY = -1

NUM_INTRUDERS = 5
NUM_WAYPOINTS = 1
INTRUSION_DISTANCE = 5 # NM

WAYPOINT_DISTANCE_MIN = 100 # KM
WAYPOINT_DISTANCE_MAX = 150 # KM

OBSTACLE_DISTANCE_MIN = 20 # KM
OBSTACLE_DISTANCE_MAX = 150 # KM

D_HEADING = 45 #degrees

AC_SPD = 150 # kts

NM2KM = 1.852

ACTION_FREQUENCY = 10

## for obstacles generation
NUM_OBSTACLES = np.random.randint(1,5)
print(NUM_OBSTACLES)
POLY_AREA_RANGE = (50, np.random.randint(100,200)) # In NM^2
CENTER = (51.990426702297746, 4.376124857109851) # TU Delft AE Faculty coordinates


class StaticObstacleCREnv(gym.Env):
    """ 
    Static Obstacle Conflict Resolution Environment

    TODO:
    - look at adding waypoints instead of staying straight
    """
    metadata = {"render_modes": ["rgb_array","human"], "render_fps": 120}

    def __init__(self, render_mode=None):
        self.window_width = 512 # pixels
        self.window_height = 512 # pixels
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
                "sin_drift": spaces.Box(-np.inf, np.inf, shape = (NUM_WAYPOINTS,), dtype=np.float64),
                # "restricted_area_intrusion": spaces.Box(0, 1, shape = (1,), dtype=np.float64),
                # "restricted_area__distance": spaces.Box(-np.inf, np.inf, shape = (n_obstacle_vertices,), dtype=np.float64),
                # "cos_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (n_obstacle_vertices,), dtype=np.float64),
                # "sin_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (n_obstacle_vertices,), dtype=np.float64)
            }
        )
       
        self.action_space = spaces.Dict(
            {
                "heading": spaces.Box(-1, 1, shape=(1,), dtype=np.float64),
                "speed": spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 5;FF')

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()

        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD)

        self._generate_other_aircraft()
        self._generate_obstacles()

        
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

    def _generate_other_aircraft(self, acid = 'KL001'):
        # for n_other_ac CRE

        # check IF AC INSIDE 


        # This originally was _generate_conflicts
        # this creates aircraft that are explicitely in conflict with the agent. 
        target_idx = bs.traf.id2idx(acid)
        for i in range(NUM_INTRUDERS):
            dpsi = np.random.randint(45,315) # Conflict angle (angle between tracks of ownship and intruder) (deg)
            cpa = np.random.randint(0,INTRUSION_DISTANCE) #Predicted distance at closest point of approach (NM)
            tlosh = np.random.randint(100,1000) 
            bs.traf.creconfs(acid=f'{i}',actype="A320",targetidx=target_idx,dpsi=dpsi,dcpa=cpa,tlosh=tlosh)

    def _generate_polygon(self, centre):
        
        R = np.sqrt(POLY_AREA_RANGE[1] / np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        #self.poly_area = p_area
        
        while p_area < POLY_AREA_RANGE[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        
        #self.poly_points = p # Polygon vertices are saved in terms of NM
        p = [fn.nm_to_latlong(centre, point) for point in p] # Convert to lat/long coordinateS
        
        # points = [coord for point in p for coord in point] # Flatten the list of points
        #bs.tools.areafilter.defineArea(self.poly_name, 'POLY', points)
        return p_area, p
    
    def _generate_obstacles(self):
        self.obstacle_names = []
        self.obstacle_vertices = []
        #self._generate_waypoint(num_waypoints = NUM_OBSTACLES)
        obst_centre_lat, obst_centre_lon = self._generate_coordinates_centre_obstacles(num_obstacles = NUM_OBSTACLES)

        for i in range(NUM_OBSTACLES):
            centre_obst = (obst_centre_lat[i], obst_centre_lon[i])
            p_area, p = self._generate_polygon(centre_obst)

            # R = np.sqrt(POLY_AREA_RANGE[1] / np.pi)
            # p = [fn.random_point_on_circle(R) for _ in range(3)] # 3 random points to start building the polygon
            # p = fn.sort_points_clockwise(p)
            # p_area = fn.polygon_area(p)
            # # self.poly_area = p_area
            
            # while p_area < POLY_AREA_RANGE[0]:
            #     p.append(fn.random_point_on_circle(R))
            #     p = fn.sort_points_clockwise(p)
            #     p_area = fn.polygon_area(p)
            
            # self.poly_points = p # Polygon vertices are saved in terms of NM
            
            # p = [fn.nm_to_latlong(CENTER, point) for point in p] # Convert to lat/long coordinateS
            
            points = [coord for point in p for coord in point] # Flatten the list of points
            poly_name = 'restricted_area_' + str(i+1)

            bs.tools.areafilter.defineArea(poly_name, 'POLY', points)
            self.obstacle_names.append(poly_name)
            self.obstacle_vertices.append(p)
            # import code
            # code.interact(local=locals())

    # original _generate_waypoints function from horizotal_cr_env
    def _generate_waypoint(self, acid = 'KL001'):
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


    # Modified _generate_waypoints function from horizotal_cr_env to avoid using a global variable for the number of obstacles. 
    # def _generate_waypoint(self, acid = 'KL001', num_waypoints = NUM_WAYPOINTS):
    #     self.wpt_lat = []
    #     self.wpt_lon = []
    #     self.wpt_reach = []
    #     for i in range(num_waypoints):
    #         wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
    #         wpt_hdg_init = 0 # always generating waypoints straight ahead?

    #         ac_idx = bs.traf.id2idx(acid)

    #         wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)    
    #         self.wpt_lat.append(wpt_lat)
    #         self.wpt_lon.append(wpt_lon)
    #         self.wpt_reach.append(0)

    def _generate_coordinates_centre_obstacles(self, acid = 'KL001', num_obstacles = NUM_OBSTACLES):
        self.obstacle_lat = []
        self.obstacle_lon = []
        # self.wpt_reach = []
        for i in range(num_obstacles):
            obstacle_dis_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            ac_idx = bs.traf.id2idx(acid)

            obstacle_lat, obstacle_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_lat.append(obstacle_lat)
            self.obstacle_lon.append(obstacle_lon)
            # self.wpt_reach.append(0)

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

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
                # add observations on obstacles
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
        drift_reward = self._check_drift()
        intrusion_reward = self._check_intrusion()

        total_reward = reach_reward + drift_reward + intrusion_reward

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
        return abs(np.deg2rad(self.drift[0])) * DRIFT_PENALTY

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        for i in range(NUM_INTRUDERS):
            int_idx = i+1
            _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
            if int_dis < INTRUSION_DISTANCE:
                reward += AC_INTRUSION_PENALTY
        
        return reward
    
    def _get_action(self,action):
            


            action = self.ac_hdg + action * D_HEADING

            bs.stack.stack(f"HDG KL001 {action[0]}")




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
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.cos(np.deg2rad(self.ac_hdg)) * ac_length)/max_distance)*self.window_width
        heading_end_y = ((np.sin(np.deg2rad(self.ac_hdg)) * ac_length)/max_distance)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (self.window_width/2-heading_end_x/2,self.window_height/2+heading_end_y/2),
            ((self.window_width/2)+heading_end_x/2,(self.window_height/2)-heading_end_y/2),
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

        # draw intruders
        ac_length = 3

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
                width = 4
            )

            # draw heading line
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
