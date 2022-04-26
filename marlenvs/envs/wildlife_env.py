from marlenvs.envs.mmdp import MMDP

from gym import spaces

import numpy as np
import random
import itertools
import copy
from symmetrizer.ops.wildlife_ops import normalize


class WildlifeEnv(MMDP):
    def __init__(self, n_agents=3, w=7, h=7, reward_type=0.,
                 inn=0):
        """
        """
        self.p = 0.8
        self.max_time = 100
        self.n_agents = n_agents
        self.capt = 0.
        self.fcapt = 0.
        self.colliders = 0.

        self.reward_type = reward_type
        self._action_space = spaces.MultiDiscrete([5 for _ in
                                                   range(self.n_agents)])
        self._action_space.n = 5 * self.n_agents

        self._action_space.decentralized = True
        self._action_space.n_actions = 5
        self._action_space.n_agents = self.n_agents
        self.act_prod = list(itertools.product(*[[0, 1, 2, 3, 4] for _ in
                                                 range(self.n_agents)]))
        self.action_dict = {0: [0, 0],   # rest
                            1: [1, 0],   # up
                            2: [0, 1],   # right
                            3: [-1, 0],  # down
                            4: [0, -1]}  # left

        self.obstacles = []
        self.w = w
        self.h = h
        self.base_coords = list(itertools.product(np.arange(self.h),
                                                  np.arange(self.w)))
        self.states = self.get_possible_coords(obstacle_list=self.obstacles)

        self._observation_space = spaces.Box(low=0., high=1.,
                                             shape=(1, self.w*3, self.h*3))

    def reset(self):
        """
        """
        self.t = 0.
        self.capt = 0.
        self.fcapt = 0.
        self.colliders = 0.
        self.global_state = [np.array(x) for x
                             in random.sample(self.states, self.n_agents+1)]
        self.local_states = []
        return self.render(self.global_state, self.local_states)

    def render(self, global_state, local_states):
        """
        """
        local_grids = np.zeros((self.n_agents, self.h*3, self.w*3))
        for i in range(self.n_agents):
            agent_loc = global_state[i]
            prey_loc = global_state[-1]

            agent_loc, prey_loc = self.egocentric_view(agent_loc, prey_loc)

            x, y = agent_loc
            p_x, p_y = self.wrap_edges_full(prey_loc[0], prey_loc[1])

            xp = 3*x
            yp = 3*y
            xc = 3*p_x
            yc = 3*p_y

            local_grids[i, xp:xp+3, yp:yp+3] = self.agent_img()
            local_grids[i, xc:xc+3, yc:yc+3] = self.prey_img()

        gs = [np.expand_dims(normalize(g_s), axis=0)
              for g_s in global_state[:-1]]
        locations = np.zeros((1, self.h*3, self.w*3))
        for i, gs in enumerate(global_state[:-1]):
            locations[0][i][0:2] = gs
        return np.concatenate([local_grids, locations], axis=0)

    def agent_img(self):
        """
        """
        img = np.zeros((3, 3))
        img[1] = 1
        img[:, 1] = 1
        return img

    def prey_img(self):
        """
        """
        img = np.zeros((3, 3))
        img[1][1] = 1
        img[0][0] = 1
        img[2][0] = 1
        img[0][2] = 1
        img[2][2] = 1
        return img

    def get_possible_coords(self, obstacle_list=[]):
        """
        Initialize a list of possible coordinates
        Outputs
        -------
        coords : list of possible state coordinates
        """
        coords = copy.deepcopy(self.base_coords)
        for obstacle in obstacle_list:
            for coord in coords:
                if np.allclose(obstacle, np.array(coord)):
                    coords.remove(coord)
        return coords

    def remove_dups(self, locations):
        """
        """
        new_locs = []
        for loc in locations:
            for loc2 in new_locs:
                if np.allclose(loc, loc2):
                    break
            new_locs.append(loc)
        return new_locs

    def initialize_map(self, height, width, obstacle_list=[]):
        """
        Initialize a grid representing the environment's map
        Outputs
        -------
        grid : numpy array containing ones for obstacles, zeros for empty space
        """
        grid = np.zeros((1, height, width))
        for obstacle in obstacle_list:
            x, y = obstacle
            grid[0][x][y] = 1
        return grid

    def local_effect(self, global_state, local_states, global_action):
        """
        Compute local effects of the global action on the global and local
        states. E.g. moving an agent from one coordinate to the other

        Outputs
        -------
        global_state_next : the new global state
        local_states_next : the new list of local states
        rewards      : the list of rewards
        """
        global_state_next = []
        rewards = []
        if "int" in str(type(global_action)) or len(global_action.shape) == 0:
            joint_action = self.act_prod[global_action]
        else:
            joint_action = global_action
        for i in range(self.n_agents):
            s_i = global_state[i]
            a_i = self.action_dict[joint_action[i]]
            local_s_i = []

            s_i_next = self.local_transition(s_i, a_i)

            r_i = self.local_reward(s_i, a_i, s_i_next, local_s_i)

            global_state_next.append(s_i_next)
            rewards.append(r_i)
        prey_act = self.get_prey_action()
        next_prey_state = self.local_transition(global_state[-1], prey_act)
        global_state_next.append(next_prey_state)
        return global_state_next, local_states, rewards

    def get_prey_action(self):
        """
        Randomly sample an action for the prey agent
        """
        r = random.random()
        if r < self.p:
            a = np.random.randint(1, 5)
            prey_move = self.action_dict[a]
        else:
            prey_move = self.action_dict[0]
        return prey_move

    def global_effect(self, n_global_state, global_state, local_states,
                      global_action, rewards):
        """
        Compute global/interaction effects of the global action on the global
        and local states. E.g. agents bumping into each other.

        Outputs
        -------
        global_state : the new global state
        local_states : the new list of local states
        rewards      : the list of rewards
        """
        self.t += 1
        prey_loc = global_state[-1]
        # Check if agents bump into each other
        colliders, n_global_state = self.check_collisions(n_global_state,
                                                          global_state)
        # Check if any agents are on the prey, removing doubles
        # (so this can be max 1 agent, otherwise it is a collider)
        prey_lander = self.check_prey_on(prey_loc, n_global_state, colliders)
        # Check if any agents are adjacent to prey, removing doubles
        prey_adjacents = self.check_prey_adjacent(prey_loc, n_global_state,
                                                  local_states, colliders)
        # Get new location for colliders, making sure they land on empty
        # coordinates and not on top of other agents or the prey
        rewards, colliders, done = self.update_rewards(rewards, colliders,
                                                       prey_lander,
                                                       prey_adjacents)
        return n_global_state, local_states, rewards, done

    def check_done(self, global_state, local_states, rewards):
        """
        """
        if self.t == self.max_time:
            return True
        return False

    def check_prey_adjacent(self, prey_loc, global_state, local_states,
                            colliders):
        """
        """
        prey_adjacents = []
        for i in range(self.n_agents):
            if self.adjacent_to(prey_loc, global_state[i]):
                if i not in colliders:
                    prey_adjacents.append(i)
        return prey_adjacents

    def adjacent_to(self, loc1, loc2):
        """
        """
        diff = abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1])
        if diff == 1:
            return True
        return False

    def check_collisions(self, n_global_state, global_state):
        """
        """
        colliders = []
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    if np.allclose(n_global_state[i], n_global_state[j]):
                        colliders.append(i)
                        colliders.append(j)
        colliders = list(set(colliders))
        self.colliders += len(colliders)

        for k in colliders:
            n_global_state[k] = global_state[k]
        return colliders, n_global_state

    def check_prey_on(self, prey_loc, global_state, colliders):
        """
        """
        prey_lander = []
        for i in range(self.n_agents):
            if np.allclose(prey_loc, global_state[i]):
                prey_lander.append(i)
        if len(prey_lander) == 1:
            return prey_lander[0]
        else:
            return None

    def update_rewards(self, rewards, colliders, prey_lander, prey_adjacents):
        """
        Update rewards based on interaction effects e.g. colliding agents,
        whether or not the prey is captured

        Outputs:
        -------
        rewards : list of agent rewards
        """
        done = False
        if self.reward_type == 1:
            for i in colliders:
                rewards[i] -= 0.25
        n_adjacent_agents = len(prey_adjacents)
        if prey_lander is not None:
            if n_adjacent_agents > 0:
                self.capt += 1.
                done = True
                rewards[prey_lander] += 1.
                for j in prey_adjacents:
                    rewards[j] += 1.
            else:
                self.fcapt += 1.
                if self.reward_type == 1:
                    rewards[prey_lander] -= 0.5
        return rewards, colliders, done

    def local_reward(self, s_i, a_i, s_i_next, local_s_i_next):
        """
        Compute direct rewards without interaction effects

        Outputs
        -------
        reward : scalar reward value
        """
        return -0.05/self.n_agents

    def local_transition(self, s_i, a_i):
        """
        Compute direct transition without interaction effects
        Outputs
        -------
        s_i_next : next state
        """
        new_location = self.move_agent(s_i, a_i)
        s_i_next = np.array(new_location)
        return s_i_next

    def move_agent(self, location, move):
        """Update agent location"""
        x, y = location
        n_x = move[0] + x
        n_y = move[1] + y
        n_x, n_y = self.wrap_edges_full(n_x, n_y)
        new_location = [n_x, n_y]
        return new_location

    def wrap_edges(self, x, y):
        """
        Toroidal grid
        """
        if x == -1:
            n_x = self.h-1
        elif x == self.h:
            n_x = 0
        else:
            n_x = x
        if y == -1:
            n_y = self.w-1
        elif y == self.w:
            n_y = 0
        else:
            n_y = y
        return n_x, n_y

    def wrap_edges_full(self, x, y):
        """
        """
        if x < 0:
            n_x = self.h+x
        elif x >= self.h:
            diff = x - self.h
            n_x = diff
        else:
            n_x = x
        if y < 0:
            n_y = self.w+y
        elif y >= self.w:
            diff = y - self.w
            n_y = diff
        else:
            n_y = y
        return n_x, n_y

    def egocentric_view(self, agent_loc, prey_loc):
        m_x = int(((self.h-1)/2))
        m_y = int(((self.w-1)/2))
        ag_x, ag_y = agent_loc

        d_x = m_x - ag_x
        d_y = m_y - ag_y

        p_x, p_y = prey_loc

        n_x = p_x + d_x
        n_y = p_y + d_y
        return [m_x, m_y], [n_x, n_y]
