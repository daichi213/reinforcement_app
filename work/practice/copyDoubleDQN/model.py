from os import name
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class Qmodel:
    def __init__(self, state_num, actions_list):
        self.state_num = state_num
        self.actions_len = len(actions_list)

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
        