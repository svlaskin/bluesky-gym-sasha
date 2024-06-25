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

NUM_INTRUDERS = 5
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
NUM_OBSTACLES = 10 #np.random.randint(1,5)
NUM_OTHER_AIRCRAFT = 5

## number of waypoints coincides with the number of destinations for each aircraft (actor and all other aircraft)
NUM_WAYPOINTS = NUM_OTHER_AIRCRAFT + 1

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

        self._generate_obstacles()

        self._generate_other_aircraft()

        self._generate_waypoint()

        self._path_planning()

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

        # bluesky reset?? bs.sim.reset() or bs.traf.reset()
        if terminated:
            for acid in bs.traf.id:
                idx = bs.traf.id2idx(acid)
                bs.traf.delete(idx)

        return observation, reward, terminated, False, info

    def _generate_other_aircraft(self, num_other_aircraft = NUM_OTHER_AIRCRAFT):
        self.other_aircraft_names = []
        for i in range(num_other_aircraft): 
            other_aircraft_name = 'ac_' + str(i+1)
            self.other_aircraft_names.append(other_aircraft_name)
            inside = True
            loop_counter = 0
            # check if aircraft is is created inside obstacle
            while inside:
                loop_counter+= 1
                bs.traf.cre(acid=other_aircraft_name,actype="A320",acspd=AC_SPD)
                ac_idx = bs.traf.id2idx(other_aircraft_name)
                for j in range(NUM_OBSTACLES):

                    # shapetemp = bs.tools.areafilter.basic_shapes[self.obstacle_names[j]]
                    inside_temp = bs.tools.areafilter.checkInside(self.obstacle_names[j], bs.traf.lat, bs.traf.lon, bs.traf.alt)
                    inside = inside_temp[ac_idx]
                    # import code
                    # code.interact(local = locals())
                if loop_counter > 1000:
                    raise Exception("No aircraft can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

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
        self._generate_coordinates_centre_obstacles(num_obstacles = NUM_OBSTACLES)

        for i in range(NUM_OBSTACLES):
            centre_obst = (self.obstacle_lat[i], self.obstacle_lon[i])
            p_area, p = self._generate_polygon(centre_obst)
            
            points = [coord for point in p for coord in point] # Flatten the list of points

            poly_name = 'restricted_area_' + str(i+1)

            bs.tools.areafilter.defineArea(poly_name, 'POLY', points)
            self.obstacle_names.append(poly_name)

            obstacle_vertices_coordinates = []
            for k in range(0,len(points),2):
                obstacle_vertices_coordinates.append([points[k], points[k+1]])
            
            self.obstacle_vertices.append(obstacle_vertices_coordinates)


    # original _generate_waypoints function from horizotal_cr_env
    def _generate_waypoint(self, acid = 'KL001'):
        self.wpt_lat = []
        self.wpt_lon = []
        self.wpt_reach = []
        for i in range(NUM_WAYPOINTS):
            
            if i == 0:
                ac_idx = bs.traf.id2idx(acid)
            else:
                ac_idx = bs.traf.id2idx(self.other_aircraft_names[i-1])
            

            inside = True
            loop_counter = 0
            while inside:
                loop_counter+= 1
                wpt_dis_init = np.random.randint(WAYPOINT_DISTANCE_MIN, WAYPOINT_DISTANCE_MAX)
                wpt_hdg_init = np.random.randint(0, 360)
                wpt_lat, wpt_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], wpt_dis_init, wpt_hdg_init)

                # working around the bug in bluesky that gives a ValueError when checkInside is used to check a single element
                wpt_lat_array = np.array([wpt_lat, wpt_lat])
                wpt_lon_array = np.array([wpt_lon, wpt_lon])
                ac_idx_alt_array = np.array([bs.traf.alt[ac_idx], bs.traf.alt[ac_idx]])
                for j in range(NUM_OBSTACLES):

                    # shapetemp = bs.tools.areafilter.basic_shapes[self.obstacle_names[j]]

                    inside_temp = bs.tools.areafilter.checkInside(self.obstacle_names[j], wpt_lat_array, wpt_lon_array, ac_idx_alt_array)
                    inside = inside_temp[0]
                    # import code
                    # code.interact(local = locals())
                if loop_counter > 1000:
                    raise Exception("No aircraft can be generated outside the obstacles. Check the parameters of the obstacles in the definition of the scenario.")

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
        
        for i in range(num_obstacles):
            obstacle_dis_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            obstacle_hdg_from_reference = np.random.randint(OBSTACLE_DISTANCE_MIN, OBSTACLE_DISTANCE_MAX)
            ac_idx = bs.traf.id2idx(acid)

            obstacle_lat, obstacle_lon = fn.get_point_at_distance(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], obstacle_dis_from_reference, obstacle_hdg_from_reference)    
            self.obstacle_lat.append(obstacle_lat)
            self.obstacle_lon.append(obstacle_lon)


    def _path_planning(self, num_other_aircraft = NUM_OTHER_AIRCRAFT):
        import pickle

        # obj0, obj1, obj2 are created here...

        # Saving the objects:
        # with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        #     obj0 = self.other_aircraft_names
        #     obj1 = bs.traf.lat
        #     obj2 = bs.traf.lon
        #     obj3 = bs.traf.alt
        #     obj4 = bs.traf.tas
        #     obj5 = self.wpt_lat
        #     obj6 = self.wpt_lon
        #     obj7 = self.obstacle_vertices
        #     pickle.dump([obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7], f)

        # Getting back the objects:
        # with open('objs-bug.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        #     obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7 = pickle.load(f)

        
        self.planned_path_other_aircraft = []

        for i in range(num_other_aircraft): 
            ac_idx = bs.traf.id2idx(self.other_aircraft_names[i])
            planned_path_other_aircraft = path_plan.det_path_planning(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], bs.traf.alt[ac_idx], bs.traf.tas[ac_idx]/kts, self.wpt_lat[i+1], self.wpt_lon[i+1], self.obstacle_vertices)
            # i = 4
            # ac_idx = bs.traf.id2idx(obj0[i])
            # planned_path_other_aircraft = path_plan.det_path_planning(obj1[ac_idx], obj2[ac_idx], obj3[ac_idx], obj4[ac_idx]/kts, obj5[i+1], obj6[i+1], obj7)
            # import code
            # code.interact(local= locals())
            # red(obj5[i+1])
            # green(obj5[i+1])
            self.planned_path_other_aircraft.append(planned_path_other_aircraft)
        
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
