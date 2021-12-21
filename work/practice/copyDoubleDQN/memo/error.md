# Error集

## 関数のデフォルト引数の順番

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

