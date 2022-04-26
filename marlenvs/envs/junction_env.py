import gym
from gym import spaces
import copy
from collections import deque, defaultdict
import numpy as np
import random
import itertools
import copy
from rlpyt.spaces.composite import Composite


class JunctionEnv(gym.Env):
    def __init__(self, n_agents=4, w=7, h=7, car_prob=0.1, gen_time_steps=100,
                 inn=0, seed=0, delay=0):
        """
        """
        self.car_prob = car_prob
        self.max_time = 499
        self.delay = 0.
        self.gen_time_steps = gen_time_steps
        self.n_agents = n_agents

        self._action_space = spaces.MultiDiscrete([2 for _ in
                                                   range(self.n_agents)])
        self._action_space.n = 2 * self.n_agents

        self._action_space.decentralized = True
        self.split_states = True
        self._action_space.n_actions = 2
        self._action_space.n_agents = self.n_agents
        self.act_prod = list(itertools.product(*[[0, 1] for _ in
                                                 range(self.n_agents)]))
        self.action_dict = {0: 'rgrg', 1: 'grgr'}
        self.action_space = self._action_space

        self.w = w*2+1
        self.h = h*2+1

        self._observation_space = spaces.Box(low=0., high=1.,
                                             shape=(1, self.w, self.h))
        self.observation_space = self._observation_space
        self.routes = [0, 1, 2, 3, 4, 5, 6, 7]

    def create_route(self, route_id):
        """
        """
        route_0 = [(i, 12) for i in range(15)]
        route_1 = [(10, i) for i in range(15)]
        route_2 = [(i, 4) for i in range(15)]
        route_3 = [(2, i) for i in range(15)]

        route_4 = [(12, i) for i in reversed(range(15))]
        route_5 = [(4, i) for i in reversed(range(15))]
        route_6 = [(i, 10) for i in reversed(range(15))]
        route_7 = [(i, 2) for i in reversed(range(15))]

        route_dict = {0: route_0,
                      1: route_1,
                      2: route_2,
                      3: route_3,
                      4: route_4,
                      5: route_5,
                      6: route_6,
                      7: route_7}
        self.route_direction = {0: "up",
                                1: "right",
                                2: "up",
                                3: "right",
                                4: "left",
                                5: "left",
                                6: "down",
                                7: "down"}
        route_que = deque(route_dict[route_id])
        return route_que

    def reset(self):
        """
        """
        self.state = []
        self.blocked = []
        self.total_vehicles = []
        self.time_vehicles = defaultdict(list)
        for t in range(self.gen_time_steps):
            for route_id in self.routes:
                if random.uniform(0, 1) < self.car_prob:
                    # Randomly choose one of the routes
                    route = self.create_route(route_id)
                    veh_id = f"veh_{route_id}_{t}"
                    direction = self.get_route_direction(route_id)
                    # Create Vehicle object
                    vehicle = Vehicle(route, t, veh_id, direction)
                    self.time_vehicles[t].append(vehicle)
                    self.total_vehicles.append(vehicle.get_id())
        self.traffic_lights = self.generate_traffic_lights()
        self.t = 0
        return self.render(self.state)

    def get_route_direction(self, route_id):
        """
        """
        return self.route_direction[route_id]

    def step(self, action):
        """
        """
        next_state = self.transition(action)
        reward = self.reward(next_state)
        env_info = {'wait_time': self.avg_wait_time,
                    'sim_time': self.avg_sim_time}

        done = False
        if len(self.total_vehicles) == 0:
            done = True
        if self.t == self.max_time:
            done = True

        team_r = np.array([reward, reward, reward, reward])
        team_done = [done, done, done, done]

        self.t += 1
        return self.render(next_state), team_r, team_done, env_info

    def render(self, state):
        """
        """
        s = 5
        full_state = np.zeros((3, self.w*s, self.h*s))
        for i in range(15):
            full_state[0][4*s:5*s, i*s:(i+1)*s] = self.road_img(s)
            full_state[0][2*s:3*s, i*s:(i+1)*s] = self.road_img(s)
            full_state[0][i*s:(i+1)*s, 4*s:5*s] = self.road_img(s)
            full_state[0][i*s:(i+1)*s, 2*s:3*s] = self.road_img(s)
            full_state[0][10*s:11*s, i*s:(i+1)*s] = self.road_img(s)
            full_state[0][12*s:13*s, i*s:(i+1)*s] = self.road_img(s)
            full_state[0][i*s:(i+1)*s, 10*s:11*s] = self.road_img(s)
            full_state[0][i*s:(i+1)*s, 12*s:13*s] = self.road_img(s)
        for tl in self.traffic_lights:
            for l in tl.locs:
                if tl.is_red_nodir(l):
                    full_state[1][l[0]*s:(l[0]+1)*s, l[1]*s:(l[1]+1)*s] = self.red_img(s, tl.get_loc_dir(l))
                else:
                    full_state[1][l[0]*s:(l[0]+1)*s, l[1]*s:(l[1]+1)*s] = self.green_img(s)
        for vehicle in state:
            x, y = vehicle.get_location()
            direction = vehicle.get_direction()
            full_state[2][x*s:(x+1)*s, y*s:(y+1)*s] = self.car_img(direction,
                                                                   s)
        if not self.split_states:
            return full_state
        else:
            return self.split(full_state)

    def split(self, full_state):
        """
        """
        w = 7
        h = 7
        s = 5
        locations = np.zeros((1, w*s, h*s))
        locations[:, 0, 0:2] = [-0.5, -0.5]
        locations[:, 1, 0:2] = [-0.5, 0.5]
        locations[:, 2, 0:2] = [0.5, -0.5]
        locations[:, 3, 0:2] = [0.5, 0.5]
        state_0 = full_state[:, :w*s, :h*s]
        state_1 = full_state[:, (w+1)*s:, :h*s]
        state_2 = full_state[:, :w*s, (h+1)*s:]
        state_3 = full_state[:, (w+1)*s:, (h+1)*s:]
        return np.concatenate([state_0, state_1, state_2, state_3, locations],
                              axis=0)

    def road_img(self, s):
        """
        """
        return np.ones((s, s)) * 0.5

    def car_img(self, direction, s):
        """
        """
        img = np.ones((s, s))
        if s == 5:
            img[0] = 0.1
            img[-1] = 0.1
            img[:, 0] = 0.1
            img[:, -1] = 0.1
        if direction == "up":
            img[-1, 2] = 1.
        elif direction == "left":
            img[2, 0] = 1.
        elif direction == "right":
            img[2, -1] = 1.
        elif direction == "down":
            img[0, 2] = 1.

        return img

    def red_img(self, s, direction):
        """
        """
        img = np.zeros((s, s))
        img[1:4, 1:4] = 1.
        if direction == "down":
            img[3, 3] = 0.
            img[3, 1] = 0.
            img[1, :] = 0.
        elif direction == "up":
            img[1, 1] = 0.
            img[1, 3] = 0.
            img[3, :] = 0.
        elif direction == "right":
            img[1, 3] = 0.
            img[3, 3] = 0.
            img[:, 1] = 0.
        elif direction == "left":
            img[3, 1] = 0.
            img[1, 1] = 0.
            img[:, 3] = 0.
        return img

    def green_img(self, s):
        """
        """
        img = np.zeros((s, s))
        img[1:4, 1:4] = 0.
        return img

    def is_empty(self, loc):
        """
        """
        for vehicle in self.state:
            if vehicle.current == loc:
                return False
        return True

    def is_red(self, previous_loc, loc):
        """
        """
        lights = self.get_traffic_light(loc)
        if lights is not None:
            if lights.is_red(previous_loc, loc):
                return True
        return False

    def get_traffic_light(self, loc):
        """
        """
        for tl in self.traffic_lights:
            # If there is a traffic light at this location, return it
            if loc in tl.locs:
                return tl
        return None

    def generate_traffic_lights(self):
        """
        """
        basis_locs = [(3, 3), (3, 11), (11, 3), (11, 11)]
        traffic_lights = []
        for b_loc in basis_locs:
            locs, directions = self.light_locations(b_loc)
            tl = TrafficLight(locs, directions)
            traffic_lights.append(tl)
        return traffic_lights

    def light_locations(self, loc):
        """
        """
        x, y = loc
        locations = [(x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
        directions = {locations[0]: get_dir((0, 0), (-1, -1)),
                      locations[1]: get_dir((0, 0), (-1, 1)),
                      locations[2]: get_dir((0, 0), (1, -1)),
                      locations[3]: get_dir((0, 0), (1, 1))}
        return locations, directions

    def transition(self, action):
        """
        """
        self.take_action(action)
        # Check if vehicles need to be added to simulation
        new_blocked = self.release_vehicles(self.t)

        new_state = []

        for vehicle in self.state:
            next_move = vehicle.peek_move()
            if next_move is None:
                # Vehicle has arrived, remove it from bookkeeping
                self.total_vehicles.remove(vehicle.get_id())
                continue
            vehicle.add_sim_time()

            # If there is no car where you want to go
            if self.is_empty(next_move):
                # If there is no traffic light stopping you
                prev_move = vehicle.get_previous()
                if not self.is_red(prev_move, next_move):
                    vehicle.do_move()
                    # Mark new location as blocked
                    new_blocked.append(vehicle.current)
                else:
                    # If traffic light is red, vehicle is waiting
                    vehicle.set_waiting(True)
            else:
                # If vehicle is blocked by another vehicle, it is waiting
                vehicle.set_waiting(True)
            new_state.append(vehicle)

        self.state = new_state
        self.blocked = new_blocked
        return new_state

    def reward(self, state):
        """
        """
        if len(state) == 0:
            self.avg_wait_time = 0.
            self.avg_sim_time = 0.
            return 0.
        total_wait = 0.
        total_sim = 0.
        for vehicle in state:
            total_wait += vehicle.get_wait_time()
            total_sim += vehicle.get_sim_time()
        self.avg_wait_time = total_wait/len(state)
        self.avg_sim_time = total_sim/len(state)
        return -self.avg_wait_time/1000.

    def release_vehicles(self, t):
        """
        """
        vehicles = self.time_vehicles[t]
        blocked = []
        for vehicle in vehicles:
            proposed_move = vehicle.peek_move()
            # Start states are never in front of traffic lights
            if self.is_empty(proposed_move):
                # If we can release the vehicle, place it at the start
                vehicle.do_move()
                blocked.append(vehicle.current)
                self.state.append(vehicle)
            else:
                # If entry point is blocked, try again at next time step
                self.time_vehicles[t+1].append(vehicle)
        return blocked

    def take_action(self, action):
        """
        """
        for i, a_i in enumerate(action):
            self.traffic_lights[i].take_action(a_i)


class TrafficLight:
    def __init__(self, locs, dirs):
        """
        """
        self.locs = locs
        self.dirs = {locs[0]: "left",
                     locs[1]: "up",
                     locs[2]: "down",
                     locs[3]: "right"}
        self.l_states = ['rggr', 'grrg']
        self.random_light_state(self.locs)

    def take_action(self, action):
        """
        """
        new_l_state = self.l_states[action]
        self.light_state = self.set_light_state(new_l_state)

    def is_red(self, prev, loc):
        """
        """
        loc_state = self.light_state[tuple(loc)]
        if loc_state == "r":
            direction = get_dir(prev, loc)
            if direction == self.dirs[loc]:
                return True
            else:
                return False
        return False

    def is_red_nodir(self, loc):
        loc_state = self.light_state[tuple(loc)]
        if loc_state == "r":
            return True
        return False

    def get_loc_dir(self, loc):
        return self.dirs[loc]

    def random_light_state(self, locs):
        """
        """
        l_state = random.choice(self.l_states)
        self.light_state = self.set_light_state(l_state)

    def set_light_state(self, l_state):
        """
        """
        light_state = {}
        for i, loc in enumerate(self.locs):
            light_state[tuple(loc)] = l_state[i]
        return light_state


def get_dir(prev_loc, loc):
    """
    """
    d_x = loc[0] - prev_loc[0]
    d_y = loc[1] - prev_loc[1]
    if d_x < 0:
        # [-1, 0]
        direction = "down"
    if d_x > 0:
        # [+1, 0]
        direction = "up"
    if d_y < 0:
        # [0, -1]
        direction = "right"
    if d_y > 0:
        # [0, +1]
        direction = "left"
    return direction


class Vehicle:
    def __init__(self, route, time_step, veh_id, direction):
        """
        """
        self.veh_id = veh_id
        self.route = route
        self.direction = direction
        self.previous = None
        self.current = None
        self.wait_time = 0.
        self.sim_time = 0.
        self.waiting = False

    def set_waiting(self, flag):
        # Set waiting flag
        self.waiting = flag
        if flag:
            self.add_wait_time()

    def is_waiting(self):
        # Set if vehicle is currently waiting
        return self.waiting

    def do_move(self):
        waiting = self.is_waiting()
        # Turn off waiting flag
        self.set_waiting(False)
        # Store current move in previous
        self.previous = self.current

        # If vehicle was waiting, it takes 1 time step to start driving
        if waiting:
            self.add_wait_time()
            return None

        # Pop first element off route
        if self.route:
            self.current = self.route.popleft()
        else:
            return None

    def peek_move(self):
        # Peek at first next element
        if self.route:
            return self.route[0]
        else:
            # If route is empty
            return None

    def get_location(self):
        # Get current vehicle location
        return self.current

    def get_id(self):
        # Get vehicle id
        return self.veh_id

    def get_previous(self):
        # Get previous move
        return self.previous

    def get_direction(self):
        # Get direction of vehicle
        return self.direction

    def add_wait_time(self):
        self.wait_time += 1

    def get_wait_time(self):
        return self.wait_time

    def add_sim_time(self):
        self.sim_time += 1

    def get_sim_time(self):
        return self.sim_time
