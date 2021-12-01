# DDQN実装メモ

## Tensorflow

### tensorflow.keras.layers.Dot

Dotがどのような計算を行っているかパッと分からなかったため、以下にメモした。[公式ページ](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dot)

```python
>>> numx=np.arange(10).reshape(1, 5, 2)
>>> numx
array([[[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]]])
>>> numy=np.arange(10,20).reshape(1, 2, 5)
>>> numy
array([[[10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]]])
>>> mul=Dot(axes=(2,1))([numx, numy])
>>> mul
<tf.Tensor: shape=(1, 5, 5), dtype=int64, numpy=
array([[[ 15,  16,  17,  18,  19],
        [ 65,  70,  75,  80,  85],
        [115, 124, 133, 142, 151],
        [165, 178, 191, 204, 217],
        [215, 232, 249, 266, 283]]])>


>>> numx
#          →　axis=1方向に計算を行う
array([[[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]]])
>>> numy
# axis=2（縦）方向に計算を行う
array([[[10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]]])
>>> mul=Dot(axes=(1,2))([numx, numy])
>>> mul
<tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
array([[[260, 360],
        [320, 445]]])>
```
axesに指定しているタプルの第1要素は1つ目のテンソルのどの軸方向へ演算を行うかを指定する。第2要素は2つ目のテンソルのどの軸方向へ演算を行うか指定する。たとえば、Dot(axes=(1,2))([numx, numy])であれば、以下のように演算が行われる。

                    10, 15
                    11, 16
                    12, 17  ↓
                    13, 18
                    14, 19
        →
    0, 2, 4, 6, 8   260 360
    1, 3, 5, 7, 9   320 445

### numpy.expand_dims

```python
>>> te=[[1,2,3,4,5],[6,7,8,9,0]]
>>> np.shape(te)
(2, 5)
>>> t=np.expand_dims(te, -1)
>>> np.shape(t)
(2, 5, 1)
>>> t
array([[[1],
        [2],
        [3],
        [4],
        [5]],

       [[6],
        [7],
        [8],
        [9],
        [0]]])

>>> te
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]
>>> t=np.expand_dims(te, 1)
>>> np.shape(t)
(2, 1, 5)
>>> t
array([[[1, 2, 3, 4, 5]],

       [[6, 7, 8, 9, 0]]])
>>>
```