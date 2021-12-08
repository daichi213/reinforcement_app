import numpy as np
from tensorflow.keras.layers import Input, Dense, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from util import idx2mask


class Qnetwork:

    def __init__(self,
                 dim_state,
                 actions_list,
                 gamma=0.99,
                 lr=1e-3,
                 double_mode=True):
        self.dim_state = dim_state
        self.actions_list = actions_list
        self.action_len = len(actions_list)
        self.optimizer = Adam(lr=lr)
        self.gamma = gamma
        self.double_mode = double_mode

        # メインネットワークの定義
        self.main_network = self.build_graph()
        # ターゲットネットワークの定義（行動価値関数の目標値の計算用グラフ）
        self.target_network = self.build_graph()
        # メインネットワークの訓練用のメソッド
        # self.main_network.predict_on_batch()である状態における各行動の行動価値関数を出力する
        # self.main_networkの訓練時の教師データとして最適行動価値関数の値とそれ以外を0として用意する
        # イメージ的に機械学習の分類問題の教師データと同じ形式として実装されているイメージ
        self.trainable_network = \
            self.build_trainable_graph(self.main_network)

    # Pendulum-v1の入力は3次元で、具体的にはある状態(x座標, y座標, 角速度)をとる
    def build_graph(self):
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = self.action_len * 10
        nb_dense_2 = int(
            np.sqrt(self.action_len * 10 *
                    self.dim_state * 10))

        l_input = Input(shape=(self.dim_state,),
                        name='input_state')
        l_dense_1 = Dense(nb_dense_1,
                          activation='relu',
                          name='hidden_1')(l_input)
        l_dense_2 = Dense(nb_dense_2,
                          activation='relu',
                          name='hidden_2')(l_dense_1)
        l_dense_3 = Dense(nb_dense_3,
                          activation='relu',
                          name='hidden_3')(l_dense_2)
        l_output = Dense(self.action_len,
                         activation='linear',
                         name='output')(l_dense_3)

        model = Model(inputs=[l_input],
                      outputs=[l_output])
        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss='mse')
        return model

    # 教師データが行動価値関数
    # Input ->[全状態変数(len(dim_state))), 行動変数(len(actions_list))]
    # Output->[最適な行動価値関数以外が0(len(actions_list))]
    def build_trainable_graph(self, network):
        action_mask_input = Input(
            shape=(self.action_len,), name='a_mask_inp')
        q_values = network.output
        q_values_taken_action = Dot(
            axes=-1,
            name='qs_a')([q_values, action_mask_input])
        # Modelのinputsには以下のようにして複数のInputLayerをlistとして入力とすることができる
        trainable_network = Model(
            inputs=[network.input, action_mask_input],
            outputs=q_values_taken_action)
        trainable_network.compile(
            optimizer=self.optimizer,
            loss='mse',
            metrics=['mae'])
        return trainable_network

    # target用Q関数に同期するための関数
    def sync_target_network(self, soft):
        weights = self.main_network.get_weights()
        target_weights = \
            self.target_network.get_weights()
        for idx, w in enumerate(weights):
            target_weights[idx] *= (1 - soft)
            target_weights[idx] += soft * w
        self.target_network.set_weights(target_weights)

    # ここでDNNの訓練が行われる
    def update_on_batch(self, exps):
        (state, action, reward, next_state,
         done) = zip(*exps)
        action_index = [
            # aの値が格納されているIndex番号を出力する
            self.actions_list.index(a) for a in action
        ]
        # one-hotベクトルへの変換
        action_mask = np.array([
            idx2mask(a, self.action_len)
            for a in action_index
        ])
        state = np.array(state)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array(done)

        # Q関数の計算
        # 次状態のactions_listすべての行動価値関数を計算
        next_target_q_values_batch = \
            self.target_network.predict_on_batch(next_state)
        next_q_values_batch = \
            self.main_network.predict_on_batch(next_state)

        # DQNのメインの計算部分
        # 次状態の行動価値観数の計算
        if self.double_mode:
            future_return = [
                # max Qt(S_t+1,a)の取得（計算で使用する値はTargetネットワークで計算する値のため以下のように取得している）
                next_target_q_values[np.argmax(
                    next_q_values)]
                for next_target_q_values, next_q_values
                in zip(next_target_q_values_batch,
                       next_q_values_batch)
            ]
        else:
            future_return = [
                np.max(next_q_values) for next_q_values
                in next_target_q_values_batch
            ]
        # 現時点での行動価値観数の計算
        # 各変数はバッチサイズのベクトルになっており、同じindex番号同士で以下のnステップ行動価値関数を計算する
        # (1 - done)は目的が達せられている状態で行動価値関数を0とするような計算式になっている
        y = reward + self.gamma * \
            (1 - done) * future_return
        # TD誤差の予測値とその損失を求めている
        # ここでMainNetの訓練を実施しており、ここで訓練した重みが定期的にTargetNetへ同期されるようになっている
        # 各状態と行動変数が訓練データで上で計算した行動価値観数が教師データ
        loss, td_error = \
            self.trainable_network.train_on_batch(
             [state, action_mask], np.expand_dims(y, -1))

        return loss, td_error
