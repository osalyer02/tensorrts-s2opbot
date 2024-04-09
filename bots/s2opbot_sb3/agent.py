from TensorRTS_Env import TensorRTS_GymEnv
import abc
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
import numpy as np
import random
from typing import Dict, List, Mapping, Tuple, Set


class Agent(metaclass=abc.ABCMeta):
    def __init__(self, initial_observation, action_space):
        self.previous_game_state = initial_observation
        self.action_space = action_space

    @abc.abstractmethod
    def take_turn(self, current_game_state): 
        """Pure virtual function in which an agent should return the move that they will make on this turn.

        Returns:
            str: name of the action that will be taken
        """
        pass

    @abc.abstractmethod
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None: 
        """Function which is called for the agent before the game begins.

        Args:
            is_player_one (bool): Set to true if the agent is playing as player one
            is_player_two (bool): Set to true if the agent is playing as player two
        """
        assert(is_player_one == True or is_player_two == True)

        self.is_player_one = is_player_one
        self.is_player_two = is_player_two

    @abc.abstractmethod
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        """Function which is called for the agent once the game is over.

        Args:
            did_i_win (bool): set to True if this agent won the game.
        """
        pass

class S2OPBot(Agent):
    def __init__(self, init_observation, action_space) -> None: 
        super().__init__(init_observation, action_space)
        print(os.getcwd())
        env = TensorRTS_GymEnv()
        self.model = PPO.load('./bots/s2opbot/models/0.zip',env=env)
    def take_turn(self, current_game_state):
        if self.is_player_one:
            clusters = current_game_state.features["Cluster"]
            tensors = current_game_state.features["Tensor"]
            map_values = np.zeros(self.mapsize, dtype=np.float32)
            for cluster in clusters:
                map_values[cluster[0]] = cluster[1]
            obs = {
                "map": map_values,
                "tensors": tensors
            }
            action, predicted_return = self.model.predict(obs)
            return action
        elif self.is_player_two:
            entities = current_game_state.features
            actions = current_game_state.actions
            done = current_game_state.done
            reward = current_game_state.reward

            clusters = entities["Cluster"]
            tensors = entities["Tensor"]

            opp_clusters = [[32-i-1, j] for i, j in clusters]
            opp_tensors = [[32-i-1, j, k, l] for i, j, k, l in tensors]
            opp_obs = Observation(
            entities={
            "Cluster": (
                opp_clusters,
                [("Cluster", i) for i in range(len(opp_clusters))]
            ),
            "Tensor": (
                opp_tensors,
                [("Tensor", i) for i in range(len(opp_tensors))]
            )
            },
            actions=actions,
            done=done,
            reward=reward
            )
            clusters = opp_obs.features["Cluster"]
            tensors = opp_obs.features["Tensor"]
            map_values = np.zeros(self.mapsize, dtype=np.float32)
            for cluster in clusters:
                map_values[cluster[0]] = cluster[1]
            obs = {
                "map": map_values,
                "tensors": tensors
            }
            action, predicted_return = self.model.predict(obs)
            return action
    
    def on_game_start(self, is_player_one : bool, is_player_two : bool) -> None:
        return super().on_game_start(is_player_one, is_player_two)
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation, action_space) -> Agent: 
    return S2OPBot(init_observation, action_space)
    
def display_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'Owen Salyer'

if __name__ == "__main__":
    agent = S2OPBot()
    print(agent.student_name_hook())