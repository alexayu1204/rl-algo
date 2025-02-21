import gym
from typing import List, Tuple

from exercise4.agents import DDPG
from exercise4.evaluate_ddpg import evaluate
from exercise5.train_ddpg \
    import BIPEDAL_CONFIG


CONFIG = BIPEDAL_CONFIG

if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
