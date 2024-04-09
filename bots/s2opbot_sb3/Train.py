from stable_baselines3 import PPO
import os
import random
import time
import wandb
from TensorRTS_Env import TensorRTS_GymEnv
from stable_baselines3.common.callbacks import BaseCallback
from natsort import natsorted


class WandbLoggingCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.win_count = 0
        self.loss_count = 0
        self.game_count = 0

    def _on_step(self):
            infos = self.locals['infos']  # Extracts info for all environments
            for info in infos:
                if 'game_over' in info and info['game_over']:  # Check if a game has ended
                    self.game_count += 1
                    if info.get('win', False):  # get win key from info
                        self.win_count += 1
                    if info.get('lost', False):  # get lost key from info
                        self.loss_count += 1

        # Only log after each game to avoid too frequent logging
            if 'game_over' in infos[0] and infos[0]['game_over']:
                win_rate = self.win_count / self.game_count if self.game_count else 0
                loss_rate = self.loss_count / self.game_count if self.game_count else 0
                wandb.log({
                    'win_rate': win_rate, 
                    'loss_rate': loss_rate, 
                    'win_count': self.win_count, 
                    'loss_count': self.loss_count, 
                    'games_played': self.game_count
            })
            return True
        
class TensorRTS_GymEnvCustReward(TensorRTS_GymEnv):
    #Custom step function with shaped/zerosum reward switching
    def __init__(self, mapsize: int = 32, nclusters: int = 6, ntensors: int = 2, maxdots: int = 9, enable_prinouts: bool = True, attack_speed: int = 2, e_to_m: float = 0.2, boom_factor: float = 0.2, attack_adv: float = 0.8):
        super().__init__(mapsize, nclusters, ntensors, maxdots, enable_prinouts, attack_speed, e_to_m, boom_factor, attack_adv)
        self.opponent_seed = random.randint(1,4)
        print(self.opponent_seed)
    def step(self, action):
        use_shaped_reward = False
        obs, reward, done, trunc, info = super().step(action)
        reward = 0
        if info['game_over']:
            if info['win']:
                reward = 100
            else:
                reward = -100
        else:
            if use_shaped_reward:
                for cluster in self.clusters:
                    if (self.tensors[0][0] - cluster[0] < 3 and self.tensors[0][0] - cluster[0] > -3):
                        reward += 0.1 * cluster[1]
                    elif (self.tensors[0][0] - cluster[0] < 2 and self.tensors[0][0] - cluster[0] > -2):
                        reward += 0.2 * cluster[1]
                reward += min(50,((0.25 * self.tensors[0][2]) + (0.4 * self.tensors[0][3])))
                # Factor in whether the tensor is in danger (opponent is close and has more power) or has advantage (opponent is close and has less power)
                if self.tensors[1][0] - self.tensors[0][0] < 2 and self.tensors[1][0] - self.tensors[0][0] > -2:
                    if self.tensor_power(1) > self.tensor_power(0):
                        reward -= 25
                    elif self.tensor_power(1) < self.tensor_power(0):
                        reward += 25
                elif self.tensors[1][0] - self.tensors[0][0] < 3 and self.tensors[1][0] - self.tensors[0][0] > -3:
                    if self.tensor_power(1) > self.tensor_power(0):
                        reward -= 10
                    elif self.tensor_power(1) < self.tensor_power(0):
                        reward += 10
                elif self.tensors[1][0] - self.tensors[0][0] < 4 and self.tensors[1][0] - self.tensors[0][0] > -4:
                    if self.tensor_power(1) > self.tensor_power(0):
                        reward -= 5
                    elif self.tensor_power(1) < self.tensor_power(0):
                        reward += 5
            else:
                reward = 0
        return obs, reward, done, trunc, info
    def opponent_act(self):
        action_seed = random.randint(0, 4)
        if self.opponent_seed == 1:
            return super().opponent_act()
        elif self.opponent_seed == 2:
            if (action_seed == 0):
                if self.tensors[1][2]>0 :   # Rush if possile
                    if int(self.e_to_m * self.tensors[1][2]) > 1:
                        self.tensors[1][1] = 2
                        self.tensors[1][2] -= int(self.e_to_m * self.tensors[1][2])
                        self.tensors[1][3] += int(self.e_to_m * self.tensors[1][2])
                    else:
                        self.tensors[1][2] -= 1
                        self.tensors[1][3] += 1
                        self.tensors[1][1] = 2 
            elif int(self.boom_factor * self.tensors[1][2] > 1):
                self.tensors[1][2] += int(self.boom_factor * self.tensors[1][2])
            else:
                self.tensors[1][2] += 1
            return self.observe()
        else:
            if action_seed == 0:
                for _ in range(self.attack_speed):
                # ensure that the player can't move past the edge of the map
                    if self.tensors[1][0] > 0:
                        self.tensors[1][0] -= 1
                        self.tensors[1][2] += self.collect_dots(self.tensors[1][0])
            elif action_seed == 1:
                if self.tensors[1][2]>0 :   # Rush if possile
                    if int(self.e_to_m * self.tensors[1][2]) > 1:
                        self.tensors[1][1] = 2
                        self.tensors[1][2] -= int(self.e_to_m * self.tensors[1][2])
                        self.tensors[1][3] += int(self.e_to_m * self.tensors[1][2])
                    else:
                        self.tensors[1][2] -= 1
                        self.tensors[1][3] += 1
                        self.tensors[1][1] = 2
            elif action_seed == 2:
                if self.tensors[1][0] < self.mapsize - 1:
                    self.tensors[1][0] += 1
                    self.tensors[1][2] += self.collect_dots(self.tensors[1][0])
            else:
                if int(self.boom_factor * self.tensors[1][2] > 1):
                    self.tensors[1][2] += int(self.boom_factor * self.tensors[1][2])
                else:
                    self.tensors[1][2] += 1  
            return self.observe()                         
        
    def reset(self, map_file_name="", seed=None):
        self.opponent_seed = random.randint(1,4)
        print("Opponent seed: {}".format(self.opponent_seed))
        return super().reset(map_file_name, seed)

# Initialize callback
callback = WandbLoggingCallback()

# Initialize wandb
run = wandb.init(

    project="TensorRTS",

    config={
         "architecture": "CNN",
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "algorithm": "PPO",
        "policy": "MultiInputPolicy",
        "timesteps": 2000,
        "env": "TensorRTS_Env",
        "batch_size": 64,
        "n_epochs": 10,
        "gae_lambda": 0.95,
    },
  #  sync_tensorboard=True,
)

model_root_dir = 'models'
log_root_dir = 'logs'

current_time = str(int(time.time()))

models_dir = os.path.join(model_root_dir, current_time)
log_dir = os.path.join(log_root_dir, current_time)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = TensorRTS_GymEnvCustReward(enable_prinouts=False)

model = PPO("MultiInputPolicy", env, learning_rate=wandb.config.learning_rate, verbose=1,
            n_steps=wandb.config.n_steps, batch_size=wandb.config.batch_size,
            n_epochs=wandb.config.n_epochs, gae_lambda=wandb.config.gae_lambda)

# Use natural sort to find the latest model.
list_folders = natsorted(os.listdir(model_root_dir), reverse=True)
for folder in list_folders:
    list_models = natsorted(os.listdir(os.path.join(model_root_dir, folder)), reverse=True)
    if len(list_models) > 0:
        model_name = os.path.join(model_root_dir, folder, list_models[0])
        print('Loading model from', model_name)
        model = PPO.load(path=model_name, env=env, device='cuda', tensorboard_log=log_dir)
        break

TIMESTEPS = 250000
iters = 0



model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callback)


model.save(os.path.join(models_dir, str(iters)))

env.close()

print(f"Saved model to {models_dir}/{iters}")

# [Optional] Finish the wandb run
wandb.finish()