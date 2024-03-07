# Tensor RTS Bot
## Structure
The agent is in ```./bots/s2opbot/agent.py```. The agent simply calls the most recent checkpoint (at ```./bots/s2opbot/checkpoints/```) and loads its ```state.agent``` into a new ```Agent```. The agent should work with the ```./tournament_runner.py``` (it has a ```agent_hook``` and ```student_name_hook```). \
\
Tensorboard files for training runs are at ```./s2opbot_training/```. The directory is split into subdirectories representing the early training with a shaped reward and the later training with a zero sum reward. Code for these custom rewards is in ```./TensorRTS.py```. \
\
Thanks to the given [ENN implementation](https://github.com/drchangliu/RL4SE/tree/main/enn/TensorRTS) for giving the framework for this project.
