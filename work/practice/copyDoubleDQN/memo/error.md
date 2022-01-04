# Error集

## python標準

### listの乗算

listで乗算を行うと以下のエラーが発生した。

```python
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-106-04c4966ef226> in <module>()
      8   state = next_state
      9 
---> 10 loss, td_error = network.update_networks(exps, double_mode=False)

<ipython-input-104-adeed3db70e0> in update_networks(self, exps, double_mode)
     53     else:
     54       future_returns = [q_main.max() for q_main in q_main_network_values]
---> 55     y = reward + self.gamma * future_returns * (1 - done)
     56     # loss, td_error = self.target_network.train_on_batch([state, action], y)
     57     loss, td_error = self.target_network.train_on_batch([state, action], np.expand_dims(y, -1))

TypeError: can't multiply sequence by non-int of type 'float'
```

#### 原因

listと他変数型を乗算したことが原因だった。list型とfloat型などは乗算することができないため、[listをnp.array型へ変換することで計算することができる。](https://stackoverflow.com/questions/485789/why-do-i-get-typeerror-cant-multiply-sequence-by-non-int-of-type-float)



### 関数のデフォルト引数の順番

```bash
(base) root@d7d24b07ecee:~/work/practice/copyDoubleDQN# python train.py
Traceback (most recent call last):
  File "/home/jovyan/work/practice/copyDoubleDQN/train.py", line 4, in <module>
    from model import Qmodel
  File "/home/jovyan/work/practice/copyDoubleDQN/model.py", line 51
    def update_values(self, double_mode=True, exps):
                                                  ^
SyntaxError: non-default argument follows default argument
```

関数に指定するデフォルト引数はノンデフォルト引数よりも後に持ってこなければエラーとなる。

## Tensorflow

### modelのinput

以下は目的となるTrainableNetwork(メインネットワーク訓練用のモデル)

```python
_________________________________________________________________
Model: "model_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_state (InputLayer)     [(None, 3)]               0         
_________________________________________________________________
hidden_1 (Dense)             (None, 30)                120       
_________________________________________________________________
hidden_2 (Dense)             (None, 45)                1395      
_________________________________________________________________
hidden_3 (Dense)             (None, 70)                3220      
_________________________________________________________________
output (Dense)               (None, 7)                 497       
=================================================================
```

以下は自分が実装した今回のモデル

```python
Model: "model_5"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_2 (InputLayer)           [(None, 5)]          0           []                               
                                                                                                  
 trainable_model_input (InputLa  [(None, 5)]         0           []                               
 yer)                                                                                             
                                                                                                  
 trainable_model_dot (Dot)      (None, 1)            0           ['input_2[0][0]',                
                                                                  'trainable_model_input[0][0]']  
                                                                                                  
==================================================================================================
```

レイヤーが想定通りに生成されていなかったが、原因となるコードは以下のように、modelのinputをMainNetworkのoutputとしていたことが原因だった。

```python
  def build_trainable_model(self, main_network):
    q_values = main_network.output
    masked_action = Input(shape=(self.num_action,), name="trainable_model_input")
    q_values_mask = Dot(axes=-1, name="trainable_model_dot")(
        [q_values, masked_action]
    )
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    # TODO 原因箇所
    trainable_model = Model(inputs=[q_values, masked_action], outputs=[q_values_mask])
    trainable_model.compile(self.optimizer, loss="mse", metrics=["mae"])
    trainable_model.summary()
    return trainable_model
```

### model.predict_on_batchの入力形状

```python
>state
array([ 0.72564335, -0.68807102,  0.60792235])
>state.shape
(3,)
>network.main_network.predict_on_batch(state)
...
    ValueError: Exception encountered when calling layer "model" (type Functional).
    
    Input 0 of layer "model_hidden1" is incompatible with the layer: expected min_ndim=2, found ndim=1. Full shape received: (3,)
```

predict_on_batchではshapeの次元が2以上出なければエラーとなるため、以下のようにしてNumpyArray次元を2以上にする。

```python
state = np.reshape(state, (1, len(state)))
```
