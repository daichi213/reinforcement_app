import gym
import numpy as np
import random
from model import Qmodel
from policy import EpsilonGreedyPolicy
from datetime import datetime
from util import RecordHistory,make_directory
import os

def train():
    env = gym.make('Pendulum-v1')
    warmup_steps = 10000
    max_episode = 300
    max_stem_num = 200
    batch_size = 200
    actions_list = [-1.0, 1.0]
    state = env.reset()
    memory = []
    # 行動系列をストックさせる最大メモリサイズ
    memory_size = 10000
    loss_list = []
    td_list = []
    step = 0
    total_step = 0
    interval = 10
    alpha = 0.1
    gamma = 0.1
    epsilon = 0.1
    state_num = env.env.observation_space.shape[0]
    # Record用のディレクトリ作成とその他準備
    header = [
        "num_episode", "loss", "td_error", "reward_avg"
    ]
    directory = make_directory("result")
    csv_name = datetime.now().strftime('%Y%m%d%H%M')
    csv_name = csv_name + ".csv"
    record = RecordHistory(
        os.path.join(directory, csv_name), header
    )
    record.generate_csv()

    dim_state = env.env.observation_space.shape[0]
    q_network = Qmodel(gamma, state_num, actions_list, dim_state)
    greedy_policy = EpsilonGreedyPolicy(q_network, epsilon)
    # warm up(行動自体は方策を使用せずにランダムに選択してメモリに行動系列を蓄積する)
    while True:
        action = random.choice(actions_list)
        next_state, reward, done, info = env.step([action])
        # reward clipping
        if reward < -1:
            c_reward = -1
        else:
            c_reward = 1
        memory.append((action, state, c_reward, next_state, done))
        state = next_state
        step+=1
        total_step+=1
        if step > max_stem_num:
            state = env.reset()
            step = 0
        if total_step > warmup_steps:
            break
        memory = memory[-memory_size:]
        print("warming up {:,} steps done".format(total_step))
    
    num_episode = 0
    episode_loop = True
    while episode_loop:
        episode_reward_list, loss_list, td_list = [], [], []
        state = env.reset()
        step = 0
        step_loop = True
        num_episode+=1
        while step_loop:
            # 行動の選択(epsilon-greedy)
            action, epsilon, q_values = greedy_policy.get_action(state, actions_list)
            # Q(s,a)←Q(s,a)+α(r+γmaxa′Q(s′,a′)−Q(s,a))
            next_state, reward, done, info = env.step([action])

            # reward clipping
            if reward < -1:
                c_reward = -1
            else:
                c_reward = 1

            memory.append((state, action, c_reward, next_state, done))
            episode_reward_list.append(c_reward)
            exps = random.sample(memory, batch_size)
            loss, td_error = q_network.update_values(exps)
            loss_list.append(loss)
            td_list.append(td_error)
            q_network.sync_target_network(soft=0.01)
            state = next_state
            memory = memory[-memory_size:]

            if step >= max_stem_num:
                step_loop = False
                reward_avg = np.mean(episode_reward_list)
                loss_avg = np.mean(loss_list)
                td_error_avg = np.mean(td_list)
                print("{:,}th step done reward_avg:{} loss_avg:{} td_avg:{}".format(num_episode, reward_avg, loss_avg, td_error))
                if num_episode % interval == 0:
                    model_path = os.path.join(
                        result_dir,
                        'episode_{}.h5'.format(num_episode)
                    )
                    q_network.main_network.save(model_path)
                    history = {
                        "num_episode" : num_episode,
                        "loss" : loss_avg,
                        "td_error" : td_error_avg,
                        "reward_avg" : reward_avg
                    }
                    record.add_histry(history)
        if num_episode >= max_episode:
            episode_loop = False

        env.close()
        print('trianing {:,} episodes... done.'.format(max_episode))

if __name__=='__main__':
    train()