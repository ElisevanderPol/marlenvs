import numpy as np
from collections import namedtuple
import gym
from gym import spaces
from rlpyt.envs.gym import GymSpaceWrapper, GymEnvWrapper

EnvStep = namedtuple("EnvStep", ["observation", "reward", "done", "env_info"])
EnvInfo = namedtuple("EnvInfo", [])  # Define in env file.
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class MMDP(gym.Env):

    def __init__(self):
        """
        """
        raise NotImplementedError()

    def step(self, global_action):
        """
        Run one timestep of the environment's dynamics. When end of episode is
        reached, reset() should be called to reset the environment's internal
        state.

        Inputs
        -----
        action : an action provided by the environment

        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a namedtuple containing other diagnostic information from the
        previous action
        """
        next_state, rewards, done = self.transition(self.global_state,
                                                    self.local_states,
                                                    global_action)
        env_info = {}
        if done:
            env_info['scapt'] = self.capt
            env_info['fcapt'] = self.fcapt
            env_info['colliders'] = self.colliders
        else:
            env_info['scapt'] = 0.
            env_info['fcapt'] = 0.
            env_info['colliders'] = 0.
        rewards = [sum(rewards) for _ in rewards]
        rewards = np.array(rewards)
        done = [done for _ in range(self.n_agents)]

        next_global_state, next_local_states = next_state
        self.global_state = next_global_state
        self.local_states = next_local_states
        local_renders = self.render(next_global_state, next_local_states)
        return local_renders, rewards, done, env_info

    def render(self, global_state, local_states):
        """
        Render image representation of the state

        Outputs
        -------
        local_renders : list of local observations
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Note that convention here is first n agents are predators, remaining
        are prey

        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is
        assumed to be 0.)
        """
        raise NotImplementedError()

    def transition(self, global_state, local_states, global_action):
        """
        Transitions environment to next global state using global action

        Outputs
        -------
        next_state : tuple(global_state, local_state_list)
        rewards    : reward_list
        """
        n_global_state, local_states, rewards = self.local_effect(global_state,
                                                                  local_states,
                                                                  global_action)
        global_state, local_states, rewards, cdone = self.global_effect(
            n_global_state, global_state, local_states, global_action, rewards)

        time_done = self.check_done(global_state, local_states, rewards)
        if cdone or time_done:
            done = True
        elif (not cdone) and (not time_done):
            done = False
        next_state = (global_state, local_states)
        return next_state, rewards, done

    def check_done(self, global_state, local_states, rewards):
        """
        Check if the episode is finished

        Outputs
        -------
        done : boolean
        """
        raise NotImplementedError()

    def local_effect(self, global_state, local_states, global_action):
        """
        Compute local effects of the global action on the global and local
        states. E.g. moving an agent from one coordinate to the other

        Outputs
        -------
        global_state : the new global state
        local_states : the new list of local states
        rewards      : the list of rewards
        """
        raise NotImplementedError()

    def global_effect(self, global_state, local_states, global_action,
                      rewards):
        """
        Compute global/interaction effects of the global action on the global
        and local states. E.g. agents bumping into each other.

        Outputs
        -------
        global_state : the new global state
        local_states : the new list of local states
        rewards      : the list of rewards
        """
        raise NotImplementedError()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

    @property
    def horizon(self):
        """Horizon of the environment, if it has one."""
        raise NotImplementedError

    def close(self):
        """Clean up operation."""
        pass


def make(*args, info_example=None, **kwargs):
    raise NotImplementedError()
