# See tutorials at https://gymnasium.farama.org/
import abc
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Mapping, Tuple, Set


class Runtime_Failure_Exception(Exception): 
    def __init__(self, responsible_bot_is_player_one : bool, responsible_bot_is_player_two : bool, parent_exception : Exception, *args: object) -> None:
        self.responsible_bot_is_player_one = responsible_bot_is_player_one
        self.responsible_bot_is_player_two = responsible_bot_is_player_two
        self.parent_exception = parent_exception
        super().__init__(*args)

class Turn_History(): 
    def __init__(self, start_board_layout : dict, act_player_one : str = None, act_player_two : str = None) -> None: 
        self.start_board_layout : dict = start_board_layout

        if act_player_one is not None: 
            self.act_player_one : str = act_player_one
        else:
            self.act_player_one : str = None

        if act_player_two is not None: 
            self.act_player_two : str = act_player_two
        else:
            self.act_player_two : str = None

    def set_player_one_action(self, action_player_one : str) -> None: 
        self.act_player_one = action_player_one

    def set_player_two_action(self, action_player_two : str) -> None: 
        self.act_player_two = action_player_two

class Tensor_Base(metaclass=abc.ABCMeta): 
    def __init__(self,
        mapsize: int = 32,
        nclusters: int = 6,
        ntensors: int = 2,  # There could be more than 2 players in the future
        maxdots: int = 9,
        enable_prinouts : bool = False,
        attack_speed: int = 2,
        e_to_m: float = 0.2,
        boom_factor: float = 0.2,
        attack_adv: float = 0.8): 

        self.enable_printouts = enable_prinouts
        
        if self.enable_printouts:
            print(f"LinearRTS -- Mapsize: {mapsize}")

        self.mapsize = mapsize
        self.maxdots = maxdots
        self.nclusters = nclusters
        self.clusters: List[List[int]] = []  # The inner list has a size of 2 (position, number of dots).
        self.tensors:  List[List[int]] = []  # The inner list has a size of 4 (position, dimension, x, y).
        # Adjustable parameters for game balance. Default values are as given in PA4 on GitHub
        self.attack_speed = attack_speed
        self.e_to_m = e_to_m
        self.boom_factor = boom_factor
        self.attack_adv = attack_adv

        self.turn_record : list[Turn_History] = []
        self.current_turn : Turn_History = None

    def record_turn(self): 
        assert(self.current_turn is not None)
        self.turn_record.append(self.current_turn)
        self.current_turn = None

    def reset(self, map_file_name=""): 
        # Reset the state of the environment to an initial state
        self.clusters: List[List[int]] = [] #Clears clusters before generating new clusters
        self.game_step_count = 0  # How many steps in a game
        positions = set()
        while len(positions) < self.nclusters // 2:
            position, b = random.choice(
                [[position, b] for position in range(self.mapsize // 2) for b in range(1, self.maxdots)]
            )
            if position not in positions:
                positions.add(position)
                self.clusters.append([position, b])
                self.clusters.append([self.mapsize - position - 1, b])
        self.clusters.sort()
 
        position = random.randint(0, self.mapsize // 2)
        self.tensors = [[position, 1, 2, 0], [self.mapsize - position - 1, 1, 2, 0]]
        # Starting positions are added for TP calculation
        self.starts = (self.tensors[0][0], self.tensors[1][0])

    def set_state(self, clusters, tensors, initial_tensor_positions):
        # self.done = state.done
        # self.reward = state.reward
        # self.clusters = state.features["Cluster"]
        # self.tensors = state.features["Tensor"]
        # self.starts = initial_tensor_positions
        self.starts = initial_tensor_positions
        self.tensors = tensors
        self.clusters = clusters

    @abc.abstractmethod
    def observe(self): 
        pass

    def tensor_power(self, tensor_index) -> float :
        # A better tensor power calculation may be possible that doesn't depend heavily on whether the unit starts on the left or right
        if tensor_index == 0:
            f = self.tensors[tensor_index][3] * (1 + (self.tensors[tensor_index][0]-(self.starts[1]-self.starts[0])/2)/self.mapsize*self.attack_adv)
        else:
            f = self.tensors[tensor_index][3] * (1 + ((self.starts[1]-self.starts[0])/2-self.tensors[tensor_index][0])/self.mapsize*self.attack_adv)
        
        if self.enable_printouts:
            print(f"TP({tensor_index})=TP({self.tensors[tensor_index]})={f}")
        return f    

    def print_universe(self):
        for j in range(self.mapsize):
            print(f" {j%10}", end="")
        print(" #")
        position_init = 0
        for i in range(len(self.clusters)):
            for j in range(position_init, self.clusters[i][0]):
                print("  ", end="")
            print(f" {self.clusters[i][1]}", end="")
            position_init = self.clusters[i][0]+1
        for j in range(position_init, self.mapsize):
            print("  ", end="")
        print(" ##")

        position_init = 0
        for i in range(len(self.tensors)):
            for j in range(position_init, self.tensors[i][0]):
                print("  ", end="")
            print(f"{self.tensors[i][2]}", end="")
            if self.tensors[i][3]>=0:
                print(f"-{self.tensors[i][3]}", end="")
            position_init = self.tensors[i][0]+1
        for j in range(position_init, self.mapsize):
            print("  ", end="")
        print(" ##")

# The Gym Env is for sb3 or gymnasium-based agents.
class TensorRTS_GymEnv(gym.Env, Tensor_Base):

    def __init__(self, mapsize: int = 32, nclusters: int = 6, ntensors: int = 2, maxdots: int = 9, enable_prinouts: bool = True, attack_speed: int = 2, e_to_m: float = 0.2, boom_factor: float = 0.2, attack_adv: float = 0.8):
        super().__init__(mapsize, nclusters, ntensors, maxdots, enable_prinouts, attack_speed, e_to_m, boom_factor, attack_adv)
        Tensor_Base.__init__(self, mapsize, nclusters, ntensors, maxdots, enable_prinouts, attack_speed, e_to_m, boom_factor, attack_adv)
        # Define action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=maxdots, shape=(mapsize,)),
            "tensors": spaces.Box(low=0, high=255, shape=(2,4))  
                # First dimension: 2: tensor 1/2; 
                # Second dimension: 4: position, dimension of the tensor, x, y
        })

    def observe(self): 
        Tensor_Base.observe(self)
        # Generate the observation based on self.clusters and self.tensors
        map_values = np.zeros(self.mapsize, dtype=np.float32)

        for cluster in self.clusters:
            map_values[cluster[0]] = cluster[1]

        # Create observation
        # Flatten the observation components into a 1D array
        # Ensure map_values has the correct shape (assuming mapsize is defined elsewhere)
        if map_values.shape != (self.mapsize,):
            raise ValueError("map_values must have shape (mapsize,)")

        # Create the observation dictionary
        observation = {
            "map": map_values.copy(),  # Copy to avoid unintended modification
            "tensors": self.tensors.copy()  # Copy to avoid unintended modification
        }
        # # Assuming mapsize and maxdots are defined elsewhere

        # # Create empty map with appropriate shape
        # map_values = np.zeros(self.mapsize, dtype=np.float32)

        # # Create tensors array with example data (modify as needed)
        # tensors = np.array([[1, 3, 100, 50],  # Tensor 1, dimension 3, at (100, 50)
        #             [2, 2, 20, 30]], dtype=np.float32)  # Tensor 2, dimension 2, at (20, 30)

        # # Combine map and tensors into observation dictionary
        # observation = {
        #     "map": map_values.copy(),  # Copy to avoid unintended modification
        #     "tensors": tensors.copy()  # Copy to avoid unintended modification
        # }
        return observation
    
    def _get_observation(self):
        return self.observe()
    
    def _get_info(self):
        information = {
            'mapsize': self.mapsize,
            'maxdots': self.maxdots,
            'nclusters': self.nclusters
        }
        return information

    def reset(self, map_file_name="", seed=None):
        super().reset(seed=seed)
        Tensor_Base.reset(self, map_file_name)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # This game will not be truncated so it is always False
        # Return value: observation, reward, terminated, truncated, info. The last item "info" can be a dictionary containing additional information from the environment, such as debugging data or specific metrics.
        print(f"{action}")
        previous_power = self.tensor_power(0)
        # Process the action
        self.process_action(0, action)
        
        current_power = self.tensor_power(0)
        # Opponent acts
        self.opponent_act()
       
        win = self.has_player_won()
        lost = self.has_player_lost()
        print(f"Win: {win}")
        print(f"Lost: {lost}")
          # Check if game ends
        terminated = self.is_game_over()
        # Calculate rewards
        reward = self.calculate_win_reward()
        reward += self.calculate_power_reward(previous_power, current_power)
        
        info = self._get_info()
        info['win'] = win
        info['lost'] = lost
        info['game_over'] = terminated 
        
        return self.observe(), reward, terminated, False, info
    
    def process_action(self,player_index, action):
        if action == 0: # Advance
            self.advance(player_index)
        
        elif action == 1: #Retreat
            self.retreat(player_index)
        
        elif action == 2: #Rush
            self.rush(player_index)
        
        elif action == 3: #Boom
            
            self.boom(player_index)
        
    def opponent_act(self):
        opponent_index = 1# This is the rush AI.
        if self.tensors[1][2]>0 :   # Rush if possile
            if int(self.e_to_m * self.tensors[1][2]) > 1:
                self.tensors[1][1] = 2
                self.tensors[1][2] -= int(self.e_to_m * self.tensors[1][2])
                self.tensors[1][3] += int(self.e_to_m * self.tensors[1][2])
            else:
                self.tensors[1][2] -= 1
                self.tensors[1][3] += 1
                self.tensors[1][1] = 2      # the number of dimensions is now 2
        else:                       # Otherwise Advance.
            for _ in range(self.attack_speed):
                if self.tensors[1][0] > 0:
                    self.tensors[1][0] -= 1
                    self.tensors[1][2] += self.collect_dots(self.tensors[1][0])

        return self.observe()
    
    def advance(self, player_index):
        
        tensor = self.tensors[player_index]
        for _ in range(self.attack_speed):
            if(tensor[0] < self.mapsize - 1):
                tensor[0] += 1
                tensor[2] += self.collect_dots(tensor[0])
    
    def retreat(self, player_index):
        tensor = self.tensors[player_index]
        if tensor[0] > 0:
            tensor[0] -= 1
            tensor[2] += self.collect_dots(tensor[0])
    
    def boom(self, player_index):
        tensor = self.tensors[player_index]
        if tensor[2] >= 1:
            conversion_amount = max(int(self.e_to_m * tensor[2]), 1)
            tensor[1] += conversion_amount
            tensor[3] += conversion_amount
            tensor[2] -= conversion_amount
    
    def rush(self, player_index):
        tensor = self.tensors[player_index]
        if tensor[2] >= 1:
            conversion_amount = max(int(self.e_to_m * tensor[2]), 1)
            tensor[1] += conversion_amount
            tensor[3] += conversion_amount
            tensor[2] -= conversion_amount
    
    def collect_dots(self, position):
        low, high = 0, len(self.clusters) - 1
        
        while low <= high:
            mid = (low + high) // 2
            current_value = self.clusters[mid][0]
            
            if(current_value == position):
                dots = self.clusters[mid][1]
                self.clusters[mid][1] = 0
                return dots
            elif current_value < position:
                low = mid + 1
            else:
                high = mid - 1
        
        return 0
    
    def calculate_win_reward(self):
        
        if self.is_game_over():
        # Assuming self.is_game_over() returns True when the game ends,
        # and you have a method or logic to determine if the player has won.
            if self.has_player_won():  # You'll need to implement this method
                return 10  
            elif self.has_player_lost():  # You'll need to implement this method
                return 0
            else :
                return 0.5 # Draw
        else:
            return 0
     
    def calculate_power_reward(self, previous_power, current_power):
        return current_power - previous_power
    
    def is_game_over(self):
        if self.tensors[0][0] >= self.tensors[1][0]:
            return True
        return False
        
    def has_player_won(self):
         # The game is considered over if one tensor overtakes the other.
        game_over = self.is_game_over()
    
        if game_over:
            # Calculate the power for both tensors.
            player_one_power = self.tensor_power(0)
            player_two_power = self.tensor_power(1)

            # Player one wins if their power is greater than player two's power.
            return player_one_power > player_two_power
    
        # If the game is not over, no one has won yet.
        return False
    
    def has_player_lost(self):
         # Assuming the game ends when one reaches the other's starting position or any other end condition.
        game_over = self.is_game_over()
        if game_over:
            player_one_power = self.tensor_power(0)
            player_two_power = self.tensor_power(1)
            return player_one_power < player_two_power
        
        return False
    
    def render(self, mode='console'):
        returnString = " "
        divider = "\n---"

        for cluster in self.clusters:
            returnString += " | " + str(cluster)
            divider += "----"
        divider += "\n"

        returnString += divider
        print(returnString)

    def close(self):
        # Clean up when closing the environment
        pass
    
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":  # This is to train

    env = TensorRTS_GymEnv()
    # This will check the custom environment and output warnings if any are found
    check_env(env)

    episodes = 100
    for i in range(episodes):
        done = False
        obs = env.reset()
        while not done:
            env.render()
            random_action = env.action_space.sample()
            print('action:', random_action)
            obs, reward, terminated, truncated, info = env.step(random_action)
            done = terminated or truncated
            print('reward:', reward, '\n')