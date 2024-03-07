import random
from typing import Dict, List, Mapping, Tuple, Set

from entity_gym.env import *
from enn_trainer import TrainConfig, State, init_train_state, train, load_checkpoint, load_agent, RogueNetAgent

from TensorRTS import TensorRTS

import hyperstate

last = load_checkpoint('./bots/s2opbot/checkpoints')
agent = RogueNetAgent(last.state.agent)

@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=TensorRTS, agent=agent.agent)

if __name__ == "__main__":  # This is to train
    main()