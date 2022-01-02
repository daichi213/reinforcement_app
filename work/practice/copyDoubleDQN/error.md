# エラー一覧

## listの乗算

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

### 原因

listと他変数型を乗算したことが原因だった。list型とfloat型などは乗算することができないため、[listをnp.array型へ変換することで計算することができる。](https://stackoverflow.com/questions/485789/why-do-i-get-typeerror-cant-multiply-sequence-by-non-int-of-type-float)

