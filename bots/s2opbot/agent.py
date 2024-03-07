import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from enn_trainer import load_checkpoint, RogueNetAgent
from TensorRTS import Agent

class S2OPBot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space)
        last = load_checkpoint('./checkpoints')
        self.current = RogueNetAgent(last.state.agent)
        
    def take_turn(self, observation : Observation) -> Action:
        move = self.current.act(observation)
        return move
    
    def on_game_start(self) -> None:
        return super().on_game_start()
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
    def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> Agent: 
        return S2OPBot(init_observation, action_space)
    
    def student_name_hook() -> str: 
        """Provide the name of the student as a string

        Returns:
            str: Name of student
        """
        return 'Owen Salyer'