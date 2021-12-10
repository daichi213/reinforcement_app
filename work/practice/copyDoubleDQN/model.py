from os import name
import numpy as np
from tensorflow import keras
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam,Adagrad
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class Qmodel:
    def __init__(self, gamma, state_num, actions_list):
        self.state_num = state_num
        self.gamma = gamma
        self.actions_list = actions_list
        self.actions_len = len(actions_list)
        self.optimizer = Adagrad(learning_rate=0.001)
        self.main_network = self.build_graph()
        self.target_network = self.build_graph()

    def build_graph(self):
        nb_layer_1 = self.dim_state * 10
        nb_layer_3 = self.actions_len * 10
        nb_layer_2 = int(np.sqrt(self.state_num * 10 * self.actions_len * 10))

        l_input = Input(shape=(self.state_num,), name='l_input')
        layer1 = Dense(nb_layer_1, activation='relu', name='hidden1')(l_input)
        layer2 = Dense(nb_layer_2, activation='relu', name='hidden2')(layer1)
        layer3 = Dense(nb_layer_3, activation='relu', name='hidden3')(layer2)
        l_output = Dense(self.actions_len, activation='linear', name='l_output')(layer3)
        model = Model(inputs=[l_input], outputs=[l_output])
        model.summary()
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def build_trainable_graph(self, action):
        # 最適行動価値関数以外を0とした教師データを生成する
        action_mask = Input(shape=(self.actions_len,), name="a_mask_input")
        best_q = Dot(axes=-1)(self.main_network, action_mask, name="a_best_q")
        build_network = Model(inputs=[self.main_network,action_mask], outputs=[best_q])
        build_network.compile(optimizer=self.optimizer,
                                loss='mse', 
                                metrics=['mae'])
        return build_network

    def sync_target_network(self,coef):
        main_network_weights = self.main_network.get_weights()
        target_network_weights = self.target_network.get_weights()
        set_weights_for_target = np.array([target_network_weights[idx] * (1 - coef) + w * coef] for idx,w in enumerate(main_network_weights))
        self.main_network.set_weights(set_weights_for_target)

    def update_values(self, double_mode, exps):
        (state, reward, action, done, next_state) = zip(*exps)
        state = np.array(state)
        reward = np.array(reward)
        action = np.array(action)
        done = np.array(done)
        next = np.array(next)
        if double_mode:
            future_return = np.array([target_q[np.argmax(main_q)] 
                for main_q, target_q 
                    in zip(
                        self.main_network.predict_on_batch(np.array(state)), self.target_network.predict_on_batch(np.array(state))
                        )
                    ])
        else:
            future_return = np.array([self.main_network.predict_on_batch(np.array(state))])
        y = reward + self.gamma * (1 - done) * future_return
        build_network = self.build_trainable_graph()
        action_mask = [1 if a==action else 0 for a in self.actions_list]