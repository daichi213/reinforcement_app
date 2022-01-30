# GAN処理の流れ

1. 事前準備
    - 各変数の定義
        - ハイパーパラメーターの定義
            - バッチサイズ
            - Discriminatorの学習率
            - Generatorの学習率
        - 事前訓練パラメータの定義
        - 訓練パラメータの定義
        - モデルの保存パスの定義(ファイルの生成までは行っていない)
    - 環境のインスタンス化
    - Agentのインスタンス化
2. pre_train関数(事前学習関数の定義(Discriminatorを予め訓練させておくための関数))
    - 真のデータを生成するためのインスタンス生成
3. train関数(メイントレーニング実行用関数)
    - 事前学習時の重みを継承する
    - 
4. Generator学習関数の定義
5. Discriminator学習関数の定義
6. Generatorの報酬決定関数の定義
7. コマンドで呼び出された場合にpre_trainとtrain関数のみを実行するように記載

## g_train

1. batch_state, batch_action, batch_rewardを生成
2. episode分学習を行う
    1. Agentの初期化
    2. 状態変数states、行動変数actions、報酬変数rewardsの定義
    3. 最大ステップ分学習を行う
        1. Agentの行動を選択する
        2. 報酬を取得する
        3. それぞれの変数をstates, actions, rewardsへ追加する(np.concatenate使用)
    4. (ステップ学習時の最終ステップの終端の変数を削除する)
    5. batch_state, batch_action, batch_rewardにstates, actions, rewardsを追加する。batch_state = [states1, states2, ...., states_batchsize]
3. agent.generatorを更新する
4. 重みの継承を行う

## d_train

1. Agentからデータの生成を行う
2. Discriminatorの学習用データを生成
3. kerasのfit_generatorメソッドを使用して学習を実行

---

## agent.py

Generatorの定義を行っているファイル

### Agent

1. 引数のクラス変数化
2. グラフの生成
3. PreGeneratorの生成(クラス変数として_build_pre_generatorからインスタンス化)
4. Generatorの生成(Actorクラスから生成)
5. _build_pre_generatorの定義
6. 事前学習関数の定義
    - 最適化関数のインスタンス化
    - PreGeneratorのコンパイル
    - PreGeneratorのサマリー出力
    - PreGeneratorの事前学習の実行
    - PreGeneratorの事前学習した際の重みを保存する
    - 事前学習した重みをGeneratorへ継承する

### Actorクラス


---

## environment.py

Discriminatorの定義を行っているファイル

### environment(Discriminator)

---

## datagenerator.py

### DataForGenerator

1. 引数のクラス変数化
2. Generator用のデータ生成関数の定義
    - 