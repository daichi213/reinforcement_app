import gym
import numpy as np
import random
from model import Qmodel

def train():
    env = gym.make('Pendulum-v1')
    warmup_steps = 100
    max_stem_num = 200
    actions_list = [-1.0, 1.0]
    state = env.reset()
    memory = []
    step = 0
    alpha = 0.1
    gamma = 0.1
    epsilon = 0.1
    q_network = Qmodel()
    # warm up(行動自体は方策を使用せずにランダムに選択してメモリに行動系列を蓄積する)
    while True:
        action = random.choice(actions_list)
        next_state, reward, done, info = env.step([action])
        memory.append(action, state, reward)
        state = next_state
        step+=1
        if step >= max_stem_num:
            break
    while True:
        while True:
            env.reset()
            # 行動の選択(epsilon-greedy)
            if random.random() <= epsilon:
                action = random.choice(actions_list)
            else:
                action = 
            # Q(s,a)←Q(s,a)+α(r+γmaxa′Q(s′,a′)−Q(s,a))
