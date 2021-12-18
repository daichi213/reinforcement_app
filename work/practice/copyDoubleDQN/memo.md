# DDQN実装メモ

## Python標準

### with open

```python
import csv
import os

cur_dir = os.getcwd()
csv_path = os.path.join(cur_dir, "test.csv")
# ファイルを新規で開き保存する
with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(self.header)
# ファイルに対して追記する
with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(self.header)
```

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

```python
>>> numx=np.array(np.arange(0,10)).reshape(5,2)
>>> numy=np.array(np.arange(10,20)).reshape(5,2)
>>> numx
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
>>> numy
array([[10, 11],
       [12, 13],
       [14, 15],
       [16, 17],
       [18, 19]])
>>> Dot(axes=-1)([numx, numy])
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
array([[ 11],
       [ 63],
       [131],
       [215],
       [315]])>
>>>
```

        10,     11
        12,     13
        14,     15
        16,     17
        18,     19

0, 1    0*10+   1*11
2, 3    2*12+   3*13
4, 5    4*14+   5*15
6, 7    6*16+   7*17
8, 9    8*18+   9*19

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

### numpy.expand_dims

```python
>>> x=np.array(np.arange(10)).reshape(1,10)
>>> x
array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
>>> np.expand_dims(x,1).shape
(1, 1, 10)
>>> np.expand_dims(x,1)
array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]])
>>> np.expand_dims(x,-1).shape
(1, 10, 1)
>>> np.expand_dims(x,-1)
array([[[0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9]]])
```

### zip(*list)

```python
>>> memory
[(1, 2, 3, 4, 5), (6, 7, 8, 9, 10), (6, 7, 8, 9, 10), (6, 7, 8, 9, 10), (6, 7, 8, 9, 10)]
>>> [a*b*c*d*e for a,b,c,d,e in zip(*memory)]
[1296, 4802, 12288, 26244, 50000]
>>> [a*b*c*d*e for a,b,c,d,e in memory]
[120, 30240, 30240, 30240, 30240]
```

        in memory

1,2,3,4,5
6,7,8,9,10
6,7,8,9,10
6,7,8,9,10
6,7,8,9,10

    ↓   in zip(*memory)

1,6,6,6,6
2,7,7,7,7
3,8,8,8,8
4,9,9,9,9
5,10,10,10,10

zip(*memory)を使用することでlist内の要素を転置して値を取り出すことができる。