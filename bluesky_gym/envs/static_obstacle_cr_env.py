import numpy as np
import pygame

import bluesky as bs
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn
import bluesky_gym.envs.common.deterministic_path_planning as path_plan
from bluesky.tools.aero import kts

import gymnasium as gym
from gymnasium import spaces
def black(text):
    print('\033[30m', text, '\033[0m', sep='')

def red(text):
    print('\033[31m', text, '\033[0m', sep='')

def green(text):
    print('\033[32m', text, '\033[0m', sep='')

def yellow(text):
    print('\033[33m', text, '\033[0m', sep='')

def blue(text):
    print('\033[34m', text, '\033[0m', sep='')

def magenta(text):
    print('\033[35m', text, '\033[0m', sep='')

def cyan(text):
    print('\033[36m', text, '\033[0m', sep='')

def gray(text):
    print('\033[90m', text, '\033[0m', sep='')

DISTANCE_MARGIN = 5 # km
REACH_REWARD = 1 # reach set waypoint

DRIFT_PENALTY = -0.1
AC_INTRUSION_PENALTY = -1
RESTRICTED_AREA_INTRUSION_PENALTY = -1

# NUM_INTRUDERS = 5
INTRUSION_DISTANCE = 5 # NM

WAYPOINT_DISTANCE_MIN = 180 # KM
WAYPOINT_DISTANCE_MAX = 400 # KM

OBSTACLE_DISTANCE_MIN = 20 # KM
OBSTACLE_DISTANCE_MAX = 150 # KM

# OTHER_AC_DISTANCE_MIN = 50 # KM
# OTHER_AC_DISTANCE_MAX = 170 # KM

D_HEADING = 45 #degrees
D_SPEED = 20/3 # kts (check)

AC_SPD = 150 # kts
ALTITUDE = 350 # In FL

NM2KM = 1.852
MpS2Kt = 1.94384

ACTION_FREQUENCY = 10

## for obstacles generation
NUM_OBSTACLES = 10 #np.random.randint(1,5)
# NUM_OTHER_AIRCRAFT = 5

## number of waypoints coincides with the number of destinations for each aircraft (actor and all other aircraft)
# NUM_WAYPOINTS = NUM_OTHER_AIRCRAFT + 1
NUM_WAYPOINTS = 1

POLY_AREA_RANGE = (50, 1000) # In NM^2
CENTER = (51.990426702297746, 4.376124857109851) # TU Delft AE Faculty coordinates

MAX_DISTANCE = 350 # width of screen in km

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

        # Observation space should include ownship and intruder info, end destination info for ownship, relative position of obstcles in reference to ownships
        # Maybe later also have an option for CNN based intruder and obstacle info, could be interesting
        self.observation_space = spaces.Dict(
            {   
                # "intruder_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                # "intruder_cos_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                # "intruder_sin_difference_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                # "intruder_x_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                # "intruder_y_difference_speed": spaces.Box(-np.inf, np.inf, shape = (NUM_INTRUDERS,), dtype=np.float64),
                "destination_waypoint_distance": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_cos_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "destination_waypoint_sin_drift": spaces.Box(-np.inf, np.inf, shape = (1,), dtype=np.float64),
                "restricted_area_radius": spaces.Box(0, 1, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "restricted_area_distance": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES, ), dtype=np.float64),
                "cos_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES,), dtype=np.float64),
                "sin_difference_restricted_area_pos": spaces.Box(-np.inf, np.inf, shape = (NUM_OBSTACLES,), dtype=np.float64)

            }
        )
       
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # initialize bluesky as non-networked simulation node
        bs.init(mode='sim', detached=True)

        # initialize dummy screen and set correct sim speed
        bs.scr = ScreenDummy()
        bs.stack.stack('DT 1;FF')

        self.obstacle_names = []

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        bs.traf.reset()
        self.counter = 0

        # bs.tools.areafilter.deleteArea(self.poly_name)

        bs.traf.cre('KL001',actype="A320",acspd=AC_SPD, acalt=ALTITUDE)

        # defining screen coordinates
        # defining the reference point as the top left corner of the SQUARE screen
        # from the initial position of the aircraft which is set to be the centre of the screen
        ac_idx = bs.traf.id2idx('KL001')
        d = np.sqrt(2*(MAX_DISTANCE/2)**2) #KM
        lat_ref_point,lon_ref_point = bs.tools.geo.kwikpos(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], 315, d/NM2KM)
        
        self.screen_coords = [lat_ref_point,lon_ref_point]#[52.9, 2.6]

        self._generate_obstacles()

        self._generate_waypoint()

        observation = self._get_obs()

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self.counter += 1
        self._get_action(action)

        action_frequency = ACTION_FREQUENCY
        for i in range(action_frequency):
            bs.sim.step()
            if self.render_mode == "human":
                self._render_frame()
            reward, terminated = self._get_reward()
            if terminated:
                observation = self._get_obs()
                info = self._get_info()
                return observation, reward, terminated, False, info

        observation = self._get_obs()
        reward, terminated = self._get_reward()

        info = self._get_info()

        return observation, reward, terminated, False, info

    def _generate_polygon(self, centre):
        poly_area = np.random.randint(POLY_AREA_RANGE[0]*2, POLY_AREA_RANGE[1])
        R = np.sqrt(poly_area/ np.pi)
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
        return p_area, p, R
    
    def _generate_obstacles(self):
        for name in self.obstacle_names:
            bs.tools.areafilter.deleteArea(name)
        self.obstacle_names = []
        self.obstacle_vertices = []
        self.obstacle_radius = []
        #self._generate_waypoint(num_waypoints = NUM_OBSTACLES)
        self._generate_coordinates_centre_obstacles(num_obstacles = NUM_OBSTACLES)

        for i in range(NUM_OBSTACLES):
            centre_obst = (self.obstacle_centre_lat[i], self.obstacle_centre_lon[i])
            p_area, p, R = self._generate_polygon(centre_obst)
            
            points = [coord for point in p for coord in point] # Flatten the list of points

            poly_name = 'restricted_area_' + str(i+1)

            bs.tools.areafilter.defineArea(poly_name, 'POLY', points)
            self.obstacle_names.append(poly_name)

            obstacle_vertices_coordinates = []
            for k in range(0,len(points),2):
                obstacle_vertices_coordinates.append([points[k], points[k+1]])
            
            self.obstacle_vertices.append(obstacle_vertices_coordinates)
            self.obstacle_radius.append(R)


    # original _generate_waypoints function from horizotal_cr_env
    def _generate_waypoint(self, acid = 'KL001'):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        for i in range(NUM_WAYPOINTS):

            ac_idx = bs.traf.id2idx(acid)
            # if i == 0:
            #     ac_idx = bs.traf.id2idx(acid)
            # else:
            #     ac_idx = bs.traf.id2idx(self.other_aircraft_names[i-1])
            
            check_inside_var = True
            loop_counter = 0
            while check_inside_var:
                loop_counter += 1
                if i == 0:
                    wpt_dis_init = np.random.randint(100, 170)
                else:
                    wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
                wpt_hdg_init = np.random.randint(0, 360)
                wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)

                # working around the bug in bluesky that gives a ValueError when checkInside is used to check a single element
                wpt_lat_array = np.array([wpt_lat, wpt_lat])
                wpt_lon_array = np.array([wpt_lon, wpt_lon])
                ac_idx_alt_array = np.array([bs.traf.alt[ac_idx], bs.traf.alt[ac_idx]])
                inside_temp = []
                for j in range(NUM_OBSTACLES):

                    # shapetemp = bs.tools.areafilter.basic_shapes[self.obstacle_names[j]]

                    inside_temp.append(bs.tools.areafilter.checkInside(self.obstacle_names[j], wpt_lat_array, wpt_lon_array, ac_idx_alt_array)[0])
                    # print(inside_temp, loop_counter)

                check_inside_var = any(x == True for x in inside_temp)
                # red(check_inside_var)
                    # check_inside_var = True 
                    
                
                if loop_counter > 1000:
                    import code
                    code.interact(local = locals())
                    raise Exception("No waypoints can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

            self.wpt_lat.append(wpt_lat)
            self.wpt_lon.append(wpt_lon)
            self.wpt_reach.append(0)


    def _generate_coordinates_centre_obstacles(self, acid = 'KL001', num_obstacles = NUM_OBSTACLES):
        self.obstacle_centre_lat = []
        self.obstacle_centre_lon = []
        
        for i in range(num_obstacles):
            obstacle_dis_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = np.random.randint(0, 360)
            ac_idx = bs.traf.id2idx(acid)

            obstacle_centre_lat, obstacle_centre_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_centre_lat.append(obstacle_centre_lat)
            self.obstacle_centre_lon.append(obstacle_centre_lon)


    # def _path_planning(self, num_other_aircraft = NUM_OTHER_AIRCRAFT):
    #     import pickle

    #     # # Saving the objects:
    #     # with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     #     obj0 = self.other_aircraft_names
    #     #     obj1 = bs.traf.lat
    #     #     obj2 = bs.traf.lon
    #     #     obj3 = bs.traf.alt
    #     #     obj4 = bs.traf.tas
    #     #     obj5 = self.wpt_lat
    #     #     obj6 = self.wpt_lon
    #     #     obj7 = self.obstacle_vertices
    #     #     pickle.dump([obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7], f)

    #     # Getting back the objects:
    #     with open('objs-bugs-v5.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    #         obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7 = pickle.load(f)

    #     self.planned_path_other_aircraft = []

    #     for i in range(num_other_aircraft): 
    #         # ac_idx = bs.traf.id2idx(self.other_aircraft_names[i])
    #         # planned_path_other_aircraft = path_plan.det_path_planning(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.alt[ac_idx], bs.traf.tas[ac_idx]/kts, self.wpt_lat[i+1], self.wpt_lon[i+1], self.obstacle_vertices)
    #         i = 2
    #         ac_idx = bs.traf.id2idx(obj0[i])
    #         planned_path_other_aircraft = path_plan.det_path_planning(obj1[ac_idx], obj2[ac_idx], obj3[ac_idx], obj4[ac_idx]/kts, obj5[i+1], obj6[i+1], obj7)
            
    #         self.planned_path_other_aircraft.append(planned_path_other_aircraft)

    #         for element in planned_path_other_aircraft:
    #             # print(element)
    #             # import code
    #             # code.interact(local= locals())
    #             bs.stack.stack(f"ADDWPT {self.other_aircraft_names[i]} {element[0]} {element[1]}")

    def _get_obs(self):
        ac_idx = bs.traf.id2idx('KL001')

        # self.intruder_distance = []
        # self.intruder_cos_bearing = []
        # self.intruder_sin_bearing = []
        # self.intruder_x_difference_speed = []
        # self.intruder_y_difference_speed = []

        self.destination_waypoint_distance = []
        self.wpt_qdr = []
        self.destination_waypoint_cos_drift = []
        self.destination_waypoint_sin_drift = []
        self.destination_waypoint_drift = []

        # obstacles 
        self.obstacle_centre_distance = []
        self.obstacle_centre_cos_bearing = []
        self.obstacle_centre_sin_bearing = []
        
        self.ac_hdg = bs.traf.hdg[ac_idx]
        
        # '''used for debugging'''
        # # other aircraft destination waypoints
        # self.other_ac_destination_waypoint_distance = []
        # '''END used for debugging'''

        # # intruders observation
        # for i in range(NUM_INTRUDERS):
        #     int_idx = i+1
        #     int_qdr, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
        
        #     self.intruder_distance.append(int_dis * NM2KM)

        #     bearing = self.ac_hdg - int_qdr
        #     bearing = fn.bound_angle_positive_negative_180(bearing)

        #     self.intruder_cos_bearing.append(np.cos(np.deg2rad(bearing)))
        #     self.intruder_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        #     heading_difference = bs.traf.hdg[ac_idx] - bs.traf.hdg[int_idx]
        #     x_dif = - np.cos(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]
        #     y_dif = bs.traf.gs[ac_idx] - np.sin(np.deg2rad(heading_difference)) * bs.traf.gs[int_idx]

        #     self.intruder_x_difference_speed.append(x_dif)
        #     self.intruder_y_difference_speed.append(y_dif)


        # destination waypoint observation
            
        self.ac_hdg = bs.traf.hdg[ac_idx]
        self.ac_tas = bs.traf.tas[ac_idx]
        wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.wpt_lat[0], self.wpt_lon[0])
    
        self.destination_waypoint_distance.append(wpt_dis * NM2KM)
        self.wpt_qdr.append(wpt_qdr)

        drift = self.ac_hdg - wpt_qdr
        drift = fn.bound_angle_positive_negative_180(drift)

        self.destination_waypoint_drift.append(drift)
        self.destination_waypoint_cos_drift.append(np.cos(np.deg2rad(drift)))
        self.destination_waypoint_sin_drift.append(np.sin(np.deg2rad(drift)))

        # '''used for debugging'''
        # # other aircraft destination waypoints
        # for i in range(NUM_OTHER_AIRCRAFT):
        #     other_ac_idx = i+1
        #     wpt_qdr, wpt_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[other_ac_idx], bs.traf.lon[other_ac_idx], self.wpt_lat[other_ac_idx], self.wpt_lon[other_ac_idx])
        #     self.other_ac_destination_waypoint_distance.append(wpt_dis * NM2KM)
        # '''END used for debugging'''

        # obstacles observations
        
        for obs_idx in range(NUM_OBSTACLES):
            obs_centre_qdr, obs_centre_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            obs_centre_dis = obs_centre_dis * NM2KM #KM        

            bearing = self.ac_hdg - obs_centre_qdr
            bearing = fn.bound_angle_positive_negative_180(bearing)

            self.obstacle_centre_distance.append(obs_centre_dis)
            self.obstacle_centre_cos_bearing.append(np.cos(np.deg2rad(bearing)))
            self.obstacle_centre_sin_bearing.append(np.sin(np.deg2rad(bearing)))

        observation = {
                # "intruder_distance": np.array(self.intruder_distance)/WAYPOINT_DISTANCE_MAX,
                # "intruder_cos_difference_pos": np.array(self.intruder_cos_bearing),
                # "intruder_sin_difference_pos": np.array(self.intruder_sin_bearing),
                # "intruder_x_difference_speed": np.array(self.intruder_x_difference_speed)/AC_SPD,
                # "intruder_y_difference_speed": np.array(self.intruder_y_difference_speed)/AC_SPD,
                "destination_waypoint_distance": np.array(self.destination_waypoint_distance)/WAYPOINT_DISTANCE_MAX,
                "destination_waypoint_cos_drift": np.array(self.destination_waypoint_cos_drift),
                "destination_waypoint_sin_drift": np.array(self.destination_waypoint_sin_drift),
                # observations on obstacles
                "restricted_area_radius": np.array(self.obstacle_radius)/WAYPOINT_DISTANCE_MAX,
                "restricted_area_distance": np.array(self.obstacle_centre_distance)/WAYPOINT_DISTANCE_MAX,
                "cos_difference_restricted_area_pos": np.array(self.obstacle_centre_cos_bearing),
                "sin_difference_restricted_area_pos": np.array(self.obstacle_centre_sin_bearing),
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
        # intrusion_other_ac_reward = self._check_intrusion_other_ac()
        intrusion_reward, intrusion_terminate = self._check_intrusion()
        
        # total_reward = reach_reward + drift_reward + intrusion_other_ac_reward + intrusion_reward
        total_reward = reach_reward + drift_reward + intrusion_reward
        if intrusion_terminate:
            return total_reward, 1
        
        if self.wpt_reach[0] == 0:
            return total_reward, 0
        else:
            return total_reward, 1
    
    def _check_waypoint(self):
        reward = 0
        index = 0
        for distance in self.destination_waypoint_distance:
            if distance < DISTANCE_MARGIN and self.wpt_reach[index] != 1:
                self.wpt_reach[index] = 1
                reward += REACH_REWARD
                index += 1
            else:
                reward += 0
                index += 1
        
        
        # for other_ac_wpt_reach_idx in range(len(self.wpt_reach)-1):
        #     # for distance in self.other_ac_destination_waypoint_distance:            
        #     if self.other_ac_destination_waypoint_distance[other_ac_wpt_reach_idx] < DISTANCE_MARGIN and self.wpt_reach[other_ac_wpt_reach_idx+1] != 1:
        #         self.wpt_reach[other_ac_wpt_reach_idx+1] = 1
        #         # green(self.counter)

        return reward

    def _check_drift(self):
        return abs(np.deg2rad(self.destination_waypoint_drift[0])) * DRIFT_PENALTY

    # def _check_intrusion_other_ac(self):
    #     ac_idx = bs.traf.id2idx('KL001')
    #     reward = 0
    #     for i in range(NUM_INTRUDERS):
    #         int_idx = i+1
    #         _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.lat[int_idx], bs.traf.lon[int_idx])
    #         if int_dis < INTRUSION_DISTANCE:
    #             reward += AC_INTRUSION_PENALTY
        
    #     return reward

    def _check_intrusion(self):
        ac_idx = bs.traf.id2idx('KL001')
        reward = 0
        terminate = 0
        for obs_idx in range(NUM_OBSTACLES):
            
            # _, int_dis = bs.tools.geo.kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], self.obstacle_centre_lat[obs_idx], self.obstacle_centre_lon[obs_idx])
            # if int_dis < INTRUSION_DISTANCE:
            if bs.tools.areafilter.checkInside(self.obstacle_names[obs_idx], np.array([bs.traf.lat[ac_idx]]), np.array([bs.traf.lon[ac_idx]]), np.array([bs.traf.alt[ac_idx]])):
                reward += RESTRICTED_AREA_INTRUSION_PENALTY
                terminate = 1
        return reward, terminate


    def _get_action(self,action):
        dh = action[0] * D_HEADING
        dv = action[1] * D_SPEED
        heading_new = fn.bound_angle_positive_negative_180(bs.traf.hdg[bs.traf.id2idx('KL001')] + dh)
        speed_new = (bs.traf.tas[bs.traf.id2idx('KL001')] + dv) * MpS2Kt

        # print(speed_new)
        bs.stack.stack(f"HDG {'KL001'} {heading_new}")
        bs.stack.stack(f"SPD {'KL001'} {speed_new}")
        
        # action_hdg = self.ac_hdg + action[0] * D_HEADING
        # action_spd = (self.ac_tas + action[1] * D_SPEED)*MpS2Kt

        # bs.stack.stack(f"HDG KL001 {action_hdg}")
        # bs.stack.stack(f"SPD KL001 {action_spd}")

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        screen_coords = self.screen_coords

        canvas = pygame.Surface(self.window_size)
        canvas.fill((135,206,235))

        # draw ownship
        ac_idx = bs.traf.id2idx('KL001')
        ac_length = 8
        heading_end_x = ((np.sin(np.deg2rad(self.ac_hdg)) * ac_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(self.ac_hdg)) * ac_length)/MAX_DISTANCE)*self.window_width
        # print(self.window_width/2, self.window_height/2)
        # pygame.draw.line(canvas,
        #     (0,0,0),
        #     (self.window_width/2-heading_end_x/2,self.window_height/2+heading_end_y/2),
        #     ((self.window_width/2)+heading_end_x/2
        #      ,(self.window_height/2)-heading_end_y/2),
        #     width = 4
        # )

        qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[ac_idx], bs.traf.lon[ac_idx])
        dis = dis*NM2KM
        x_actor = ((np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.window_width
        y_actor = ((-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE)*self.window_width

        pygame.draw.line(canvas,
            (235, 52, 52),
            (x_actor-heading_end_x/2, y_actor),
            (x_actor+heading_end_x/2, y_actor),
            width = 4
        )

        # draw heading line
        heading_length = 50
        heading_end_x = ((np.sin(np.deg2rad(self.ac_hdg)) * heading_length)/MAX_DISTANCE)*self.window_width
        heading_end_y = ((np.cos(np.deg2rad(self.ac_hdg)) * heading_length)/MAX_DISTANCE)*self.window_width

        pygame.draw.line(canvas,
            (0,0,0),
            (x_actor,y_actor),
            (x_actor+heading_end_x, y_actor-heading_end_y),
            width = 1
        )

        # draw obstacles
        for vertices in self.obstacle_vertices:
            points = []
            for coord in vertices:
                lat_ref = coord[0]
                lon_ref = coord[1]
                qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat_ref, lon_ref)
                dis = dis*NM2KM
                x_ref = (np.sin(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                y_ref = (-np.cos(np.deg2rad(qdr))*dis)/MAX_DISTANCE*self.window_width
                points.append((x_ref, y_ref))
            pygame.draw.polygon(canvas,
                (0,0,0), points
            )

        # # draw intruders
        # ac_length = 3

        # for i in range(NUM_INTRUDERS):
        #     int_idx = i+1
        #     int_hdg = bs.traf.hdg[int_idx]
        #     heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * ac_length)/MAX_DISTANCE)*self.window_width
        #     heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * ac_length)/MAX_DISTANCE)*self.window_width

        #     int_qdr, int_dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], bs.traf.lat[int_idx], bs.traf.lon[int_idx])

        #     # determine color
        #     if int_dis < INTRUSION_DISTANCE:
        #         color = (220,20,60)
        #     else: 
        #         color = (80,80,80)

        #     x_pos = (np.sin(np.deg2rad(int_qdr))*(int_dis * NM2KM)/MAX_DISTANCE)*self.window_width
        #     y_pos = -(np.cos(np.deg2rad(int_qdr))*(int_dis * NM2KM)/MAX_DISTANCE)*self.window_height

        #     pygame.draw.line(canvas,
        #         color,
        #         (x_pos,y_pos),
        #         ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
        #         width = 4
        #     )

        #     # draw heading line
        #     heading_length = 10
        #     heading_end_x = ((np.sin(np.deg2rad(int_hdg)) * heading_length)/MAX_DISTANCE)*self.window_width
        #     heading_end_y = ((np.cos(np.deg2rad(int_hdg)) * heading_length)/MAX_DISTANCE)*self.window_width

        #     pygame.draw.line(canvas,
        #         color,
        #         (x_pos,y_pos),
        #         ((x_pos)+heading_end_x,(y_pos)-heading_end_y),
        #         width = 1
        #     )

        #     pygame.draw.circle(
        #         canvas, 
        #         color,
        #         (x_pos,y_pos),
        #         radius = (INTRUSION_DISTANCE*NM2KM/MAX_DISTANCE)*self.window_width,
        #         width = 2
        #     )

            # import code
            # code.interact(local=locals())

        # draw target waypoint
        indx = 0
        for lat, lon, reach in zip(self.wpt_lat, self.wpt_lon, self.wpt_reach):
            
            indx += 1
            qdr, dis = bs.tools.geo.kwikqdrdist(screen_coords[0], screen_coords[1], lat, lon)

            circle_x = ((np.sin(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width
            circle_y = (-(np.cos(np.deg2rad(qdr)) * dis * NM2KM)/MAX_DISTANCE)*self.window_width


            if reach:
                color = (155,155,155)
                color_actor_target = (18, 14, 120)
            else:
                color = (255,255,255)
                color_actor_target = (235, 52, 52)
            
            if indx == 1:
                color = color_actor_target

            pygame.draw.circle(
                canvas, 
                color,
                (circle_x,circle_y),
                radius = 4,
                width = 0
            )
            
            pygame.draw.circle(
                canvas, 
                color,
                (circle_x,circle_y),
                radius = (DISTANCE_MARGIN/MAX_DISTANCE)*self.window_width,
                width = 2
            )

        self.window.blit(canvas, canvas.get_rect())
        
        pygame.display.update()
        
        self.clock.tick(self.metadata["render_fps"])
        # pygame.time.wait(10**5)

    def close(self):
        pass
