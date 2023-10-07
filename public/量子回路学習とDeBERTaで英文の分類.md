---
title: 量子回路学習とDeBERTaで英文の分類
tags:
  - 自然言語処理
  - ディープラーニング
  - 量子アルゴリズム
  - PennyLane
  - 量子回路学習
private: false
updated_at: '2022-12-01T16:47:26+09:00'
id: f976dfa018a920c78426
organization_url_name: null
slide: false
ignorePublish: false
---
この記事は[BrainPad Advent Calender 2022](https://qiita.com/advent-calendar/2022/brainpad) 1日目の記事です。
今日はQuantum Circuit Learning(量子回路学習)とDeBERTaを使った英文の分類についての話をしたいと思います。
※本記事の量子計算は量子コンピュータのシミュレータを使っています。量子コンピュータは使っていません。
# はじめに
　新卒一年目のBrainPadのデータサイエンティストです。そろそろ量子計算を使いたい季節になってきましたね。ということで量子計算(量子アルゴリズム)を使って何か書こうと考えたのですが、現在実装できる量子アルゴリズムで実務レベルで古典のアルゴリズムに勝てるものは私の知る限りではありません。そこでなにか量子計算が役に立つタスクがないかと試したのがこの記事になります。
　今回は自然言語処理のタスクである文章分類(英文の分類)をkaggleでもよく使われるモデルのDeBERTaと量子アルゴリズムのQCL(量子回路学習)をHuggingFaceとPennyLane使って実装し、普通のニューラルネットワークのDeBERTaと比較します。結果は...
実装:[Google colab](https://colab.research.google.com/gist/fuyu-quant/d24e1f451cca1f7d3906b08da02aeead/deberta_qcl.ipynb)
記事に誤り等ありましたらご指摘いただけますと幸いです。
# 目次
- [1. QCL(量子回路学習)とは](#1-QCL(量子回路学習)とは)
- [2. 全体の実装](#2-全体の実装)
- [3. 結果](#3-結果)
- [4. おわりに](#4-おわりに)
- [5. 参考文献](#5-参考文献)
- [6. 論文](#6-論文)



# 1. QCL(量子回路学習)とは
　QCL(量子回路学習)は、NISQ(Noisy Intermediate Scale Quantum)(現在、および近い未来のノイズがある量子コンピュータ)で動くことを目指した量子機械学習アルゴリズムです。
論文：[Quantum Circuit Learning](https://arxiv.org/pdf/1803.00745.pdf)
### QCLについて
　ゲート型量子コンピュータの一般的な計算方法では、以下のような複数の量子ゲートを使って量子回路を組み量子ビットを変化させ最後に量子ビットを測定することで計算結果を取り出します。

<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/7541fef2-bd4a-70a4-c2b3-c96e4d7d26e9.png" width=50%>


図１. 量子回路の例
(引用元:[量子回路の可視化](https://qiskit.org/documentation/locale/ja_JP/tutorials/circuits_advanced/03_advanced_circuit_visualization.html))

　量子回路学習では、上記の量子回路に後から変更することが可能なパラメータを持つ量子ゲートを加えることで量子回路からの出力を変更できるようにします。さらにこのパラメータを出力したい値に近づけるために最適化し、量子回路で任意の関数関係を表現することを目指します。
　このような学習の方法は皆さんお馴染みの**ニューラルネットワークでの学習**と非常に類似しています。ニューラルネットワークではパラメータ付きの線形変換器と非線形変換器(活性化関数)を使ったモデルを使い訓練データの入出力関係を近似するように学習を行います。一方、量子回路学習では**複数量子ビットのテンソル積構造による非線形性**を使い、パラメータは**量子ゲートの回転操作の回転角**として表現します。
　以下が今回扱う量子回路の概略図です。左側のブロックがデータの入力を行う量子ゲートで右側のブロックがパラメータを含む量子ゲートになっています。
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/e921efd1-f96c-cbdd-48bc-dbba24770c5b.png" width=40%>
図2. 量子回路学習の量子回路
(引用元：http://docs.qulacs.org/en/latest/apply/5.2_qcl.html)

　また量子回路を学習モデルに使うことで、**重ね合わせの原理による高次元の特徴量空間が利用でき、古典のアルゴリズムに比べて効率的に表現しにくい非線形モデルを構築できる**そうです。また、**量子ゲートは全てユニタリー変換であり、ノルムが常に1であることがある種の正則化として機能し過学習が抑えられる**そうです。
関連研究には以下のようなものがあります。
[On the Expressibility and Overfitting of Quantum Circuit Learning](https://dl.acm.org/doi/pdf/10.1145/3466797)


それでは量子回路学習のための量子回路の実装方法について説明します。
### PENNYLANEについて
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/51215d10-45dc-c9c2-7da2-c578a11300d0.png" width=50%>

量子回路を組むと言っても現在では優秀なライブラリがあるおかげで簡単に実装することができます。
今回はXanadu(ザナドゥ)という会社の[PennyLane](https://pennylane.ai/qml/index.html)というPythonのライブラリを使います。他にも量子計算を試すためのライブラリは、
- [TensorFlow Quantum](https://www.tensorflow.org/quantum)(Google)
- [blueqat](https://github.com/Blueqat/Blueqat)(blueqat)
- [Qiskit](https://qiskit.org/)(IBM)
- ...

などたくさんあります。

### QCLの実装
QCLの実装としては色々な方法が提案されています。今回は３つほど量子回路学習モデルを実装します。初めにベースになるものの実装です。
```python
n_qubits = 10
num_wires = 10
n_layers = 1

# C++で書かれた高速シミュレータ
dev = qml.device("lightning.qubit", wires=num_wires)

@qml.qnode(dev, diff_method='adjoint')
def qnode(inputs, weights):
    # 入力データの埋め込みに使う量子ゲート
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # パラメータ付きの回路と量子もつれのための回路
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # 全ての量子ビットをz基底で測定する
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
```
以上で量子回路自体の実装は完了です。すごく簡単です！！
今回実装した量子回路は、以下のような構造になっています。
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/3251a21d-e41b-b01e-ca7f-478e0335ddea.png" width=60%>
これを細かく表示したのが下の図で、左から順番に以下の量子ゲートが量子ビットに作用しています。
1. RX回転ゲートで入力データを回転角度として埋め込む
1. RX,RY,RZの回転ゲートで量子ビットを回転させる、ここの回転角度がパラメータ
回転ゲートについては以下の記事などが参考になるかもしれません
[量子コンピュータの基本 - 回転ゲートについてしっかり学ぶ](https://qiita.com/ttabata/items/ff3970e0c6a4068f1f55#sailboat-%E4%BB%BB%E6%84%8F%E5%9B%9E%E8%BB%A2%E6%93%8D%E4%BD%9C%E3%81%A8%E5%9B%9E%E8%BB%A2%E8%A8%88%E7%AE%97%E7%B5%90%E6%9E%9C)
1. CNOTゲートにより量子もつれ状態を作る(隣接量子ビット間で量子もつれを作る)
1. Z基底で測定を行い出力データとする

この量子回路を**1layer QCL**とします。
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/9c1a2528-31e8-dbdc-b619-6abb78855e05.png" width=80%>
その他に今回は以下のような量子回路でも検証を行いました。
**parameter 3layer QCL**:量子回路のパラメータ部分をさらに3回繰り返したもの
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/7754cfed-a822-9c9b-312c-390594021bfa.png" width=70%>

**3layer QCL**：埋め込み層も含めて量子回路をさらに3回繰り返したもの
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/4aed89f4-2d45-fea7-9b51-ac312dc12e2a.png" width=80%>
ここまでできればあとはDeBERTaのモデルにQCLの量子回路を組み込むだけです。

# 2. 全体の実装
　今回は全体の実装をpytorchで行いました。学習の可視化にはwandbを使っています。全体の実装は以下からも確認できます。
[Google colab](https://colab.research.google.com/gist/fuyu-quant/d24e1f451cca1f7d3906b08da02aeead/deberta_qcl.ipynb)
### 主に必要なライブラリ
```
pip install datasets
pip install transformers
pip install PennyLane
pip install wandb
```
```python
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchinfo import summary

from transformers import DataCollatorWithPadding
from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW

# 量子機械学習ライブラリ
import pennylane as qml
from pennylane import numpy as np
```

### 利用したデータセット
　英語の映画レビューデータセットで、各レビューに対してネガティブかポジティブか二値のレベルが付与されています。
```python
imdb = load_dataset("imdb")
train_df = pd.DataFrame()
train_df['text'] = imdb['train']['text']
train_df['label'] = imdb['train']['label']
train_df.head()
```
以下のようなデータが含まれています。
![スクリーンショット 2022-12-01 2.38.15.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/6d4b51bd-b63a-4b5d-ae7f-5a951f16ae6a.png)

### モデル全体像
　DeBERTaとQCLを使うと書いていますがモデルの構造は単純でDeBERTaのCLSトークンの出力をQCLに入力できるサイズに全結合層でサイズを減らしQCLに入力、QCLの出力を全結合層に通して二値分類の出力にしています。
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/2e73657d-7045-5c79-af16-a60fc2c4d7f9.png" width=60%>


### モデルの実装
**それではDeBERTaにくっつけます。**
以下が今回使用したモデルで見ての通りほとんどが通常の自然言語処理のモデルになっています、先ほど実装した**qnode**が新しく追加されているだけです。
```python
class HybridModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg =cfg
        self.backbone = AutoModel.from_pretrained('microsoft/deberta-base')
        self.config = AutoConfig.from_pretrained('microsoft/deberta-base')

        self.Linear1 = nn.Linear(self.config.hidden_size, 10)
        self.Linear3 = nn.Linear(10, self.cfg.num_classes)

        # ここが量子回路になります！！
        # 先ほど実装したqnodeと重みの形状が引数として与えられています。
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, attention_mask, label = None):
        out = self.backbone(input_ids = input_ids, attention_mask = attention_mask,
                         output_hidden_states = False)
        output = out[0][:, 0, :]
        output = self.Linear1(output)
        output = self.qlayer(output)
        output = self.Linear3(output)

        if label is not None:
            loss = self.criterion(output, label)
        else:
            loss = None
        return output, loss 
```
※pennylaneでは[Keras](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html)にも量子回路を組み込むことができるそうです。
### その他
　あとは通常のディープラーニンモデルの学習を行う時の実装と同じです。Dataloaderやoptimizer,scheduler,モデルの評価のタイミングなど自由に実装してください！！ディープラーニングの分かりやすい実装方法の記事は世の中にたくさんあるので、ここではそれらの実装方法については省略します。

# 3. 結果
　学習の様子はwandbで可視化しました、学習曲線の様子は以下です。
QCLを含むモデルは圧倒的に学習が収束するのが遅かったです、精度は少しずつ良くなってはいるのですが学習は途中までで止めています。
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/8a805acc-9d81-77dc-4409-0e5bce223d60.png" width=60%>
※DeBERTa-baseは1epoch、それ以外は3epochまで学習させています。
　続いて、AUCは以下のようになりました。
ハイパラ探索などは行っていないので以下のスコアは参考程度になります。
|モデル  |AUC |
|-------|-------|
|DeBERTa-base|0.98963|
|DeBERTa-base + 1layer QCL|0.97975|
|DeBERTa-base + parameter 3layer QCL|0.96981|
|DeBERTa-base + 3layer QCL|0.96982  |

　今回の英文の分類タスクがそもそも簡単なタスクであったみたいで、どれも非常に高い精度となりました。
学習の収束の速さに関しては通常のディープラーニングモデルが圧倒的であったので量子回路学習をより早く収束させる方法を模索したいと思いました。
### 改善点
改善できそうな箇所を以下に記載しています。
- 量子回路学習の量子回路を変更する
パラメータ量子回路は以下のように他にも用意されているので変更することができます。
```python
qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
```
- optimizerの変更
- schedulerの変更
- 量子回路の組み込み方を変える
以下では並列して複数の量子回路を使う方法が提案されている
[Turning quantum nodes into Torch Layers](https://pennylane.ai/qml/demos/tutorial_qnn_module_torch.html)
- 他のデータセットで試す
など、さまざまな変更を加えることができそうです。

# 4. おわりに
　今回は量子機械学習アルゴリズムであるQCLをDeBERTaの出力部分に加えたモデルを使い、英語の文章分類のタスクで検証を行いました。結果は**QCLを使ったものでもある程度精度が出せる**ことが分かりましたが、**従来のモデルに比べて精度(正確には途中で止めたので分かりませんが...)および学習の収束速度で劣っている**結果になりました。
　予測の得意なデータと不得意なデータが通常のDeBERTaとDeBERTa+QCLのモデルで異なっていたりすれば、2つのモデルでアンサンブルすることでより精度を高めるなど量子機械学習が役に立つ可能性を見出せそうだな〜とか思いましたが、今回は間に合わず検証できませんでした。
　引き続き量子機械学習モデルを現実のデータセットに対して活用し検証をしていきたいと考えています。いつかkaggleとかで量子機械学習が活躍していたらいいですね〜

# 5. 参考文献
- [量子コンピューティング: 基本アルゴリズムから量子機械学習まで](https://www.ohmsha.co.jp/book/9784274226212/)
- [Pennylaneのチュートリアル](https://pennylane.ai/qml/demos/tutorial_qnn_module_torch.html)
- [Quantum Native Dojo - Quantum Circuit Learning](https://dojo.qulacs.org/ja/latest/notebooks/5.2_Quantum_Circuit_Learning.html)
- [量子コンピュータは「過学習」しにくい　グリッドが量子AIの研究結果を「ACM」で発表　量⼦機械学習器のVC次元を初めて確立](https://robotstart.info/2021/07/26/grid-qcvc.html)
- [(徒然に)量子回路学習の所感](https://qiita.com/notori48/items/85f1a7f559abd96b8bc8)

# 6. 論文
- [Quantum Circuit Learning](https://arxiv.org/abs/1803.00745)
量子回路学習について
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
DeBERTaの論文
- [On the Expressibility and Overfitting of Quantum Circuit
Learning](https://dl.acm.org/doi/pdf/10.1145/3466797)
量子機械学習で過学習が抑制される性質があることの実証実験と理論的解析
