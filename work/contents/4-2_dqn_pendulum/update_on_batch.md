# Model更新の流れ

1. 行動系列memoryから各変数を取得する
2. 取得した各変数をnumpy.arrayへ変形する
3. main_networkとtarget_networkそれぞれからQ関数の値を取得する(predict_on_batchメソッドを使用する)
4. DoubleDQNまたは通常のDQNを選択して最終的な行動価値関数の値を決定する
5. networkの学習を行う(train_on_batchメソッドを使用する)
