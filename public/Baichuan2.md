---
title: Baichuan2
tags:
  - LLM
  - Baichuan2
private: false
updated_at: '2023-10-06T02:09:11+09:00'
id: 6c487f9a80cb3c4e9768
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

今回はBaichuan2の紹介になります．Baichuan Intelligenceという中国の企業が開発した新しいオープンソースのLLMになります．中国語，英語，多言語において幾つかのデータセットで同じパラメータ数を持つモデルの中で最高精度を出したモデルになります．

記事に誤り等ありましたらご指摘いただけますと幸いです。

https://github.com/baichuan-inc/Baichuan2/tree/main


# 目次
- [1. Baichuan2](#1-baichuan2)
- [2. 使い方](#2.-使い方)
    - [日本語での質問](#日本語での質問)
    - [英語での質問](#英語での質問)
    - [コード生成(日本語)](#コード生成(日本語))
    - [計算](#計算)
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)

# 1. Baichuan2

ライセンス:Apache-2.0(研究では自由に使ってよく，商用利用したい場合はメールで問い合わせる必要があるそうです)
リポジトリ:https://github.com/baichuan-inc/Baichuan2/tree/main
公式サイト:https://www.baichuan-ai.com/home
論文:https://arxiv.org/abs/2309.10305

英語に特化しているモデルが多い中，中国語に特化させたモデルを開発．数学やコード生成，医療や法律などの専門的な領域で高い性能を発揮しているそうです．日本語も学習データに含まれているため日本語の出力も行うことができます．

## 公開されているモデル(2023/9/26)

- ベースモデル
    - [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
    - [Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)
- チャットモデル
    - [Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
    - [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)(今回利用したモデル)
- 量子化したチャットモデル(モデルの重みがまだ公開されていないみたいでした(2023/9/26))
    - [Baichuan2-7B-Chat-4bits](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat-4bits)
    - [Baichuan2-13B-Chat-4bits](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat-4bits)

# 2. 使い方
すぐに試したい方はData Science WikiのページまたはColabのリンクから実行してみてください

<a href="https://www.data-science-wiki.net/article?path=/nlp/text_generation(japanese)/baichuan2_13b_chat.html">
<img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/nlp/text_generation(japanese)/baichuan2_13b_chat.ipynb)

今回はBaichuan2-13B-Chatを使います．GPUは30GBほどのメモリ消費でしたのでGoogle ColabのA100(40G)などで動かせます．

必要なライブラリをインストールします．

```python
!pip install transformers_stream_generator sentencepiece bitsandbytes accelerate
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
```

モデルのダウンロード

```python
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-13B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat")
```

## 日本語での質問

```python
messages = []
messages.append({"role": "user", "content": "機械学習について教えてください"})
response = model.chat(tokenizer, messages)
print(response)

# 出力
機械学習は、人工智能の一つの分野です。その目的は、データからプログラミングによる手動の介入を必要としないアルゴリズムやモデルを作成することです。機械学習は、画像認識、音声認識、自然言語処理、推論など、さまざまな分野で活用されています。

機械学習の基盤は、数学（特に線形代数、概率論、統計学）、電子情報工学、物理学、心理学、生物学、社会科学など多くの学問が交差しています。

機械学習の主要な手法の1つは、回帰分析です。これは、2次元の平面図上の点群から直線または回帰線を引くことで、各点の特徴量（x座標、y座標）と予測したい特徴量（y）の間の関係を捉える方法です。

また、機械学習の手法の1つは、決定木（Decision Tree）です。決定木は、木の幹、枝、葉の形を擬ねたもので、データの中でどのような決定を下すかを表現するための方法です。決定木は、予測結果が理解できるような明確な形で予測結果を出します。

最も知られている機械学習手法の1つは、神经網（Neural Network）です。これは、人の神経細胞の組み合わせを参考にしたもので、複数の層が存在し、各層のノードは相手のノードと通信し合い、情報を伝達することです。神经網は、画像認識、音声認識、自然言語処理など、さまざまな分野で活用されています。

機械学習の実行には、PythonのScikit-learn、TensorFlow、Kerasなどの機械学習ライブラリが利用できます。
```

よい出力が得られました．日本語で利用する上でも十分に使える精度だと思われます．一部漢字の出力について日本語に無い自体で出力されました．

## 英語での質問

```python
messages = []
messages.append({"role": "user", "content": "Please tell me about quantum computers."})
response = model.chat(tokenizer, messages)
print(response)

# 出力
Quantum computers are a relatively new and rapidly developing technology that uses the principles of quantum mechanics to perform calculations and solve problems. They are different from traditional digital computers, which use binary bits (0s and 1s) to represent information, in that they use qubits (quantum bits) as their basic units.

Here are some key features and differences between classical and quantum computers:

1. Speed: Quantum computers are believed to be exponentially faster than classical computers for certain tasks, such as factoring large numbers or searching large databases. This is because they can exploit the superposition and entanglement properties of qubits to perform complex calculations in parallel.

2. Simultaneous calculation: In a quantum computer, all qubits can be in multiple states at the same time, allowing them to process and analyze vast amounts of data simultaneously. This is known as quantum parallelism and can significantly increase the computational power of the machine.

3. Entanglement: Quantum computers can entangle qubits together, meaning that changing the state of one qubit will instantaneously affect the state of the other qubits. This allows for more efficient and accurate computations, but also makes them more vulnerable to interference and noise.

4. Stability: Due to their delicate nature, quantum computers require extremely low temperatures and specialized environments to prevent external interference. This makes them more difficult to build and maintain than classical computers.

5. Applications: Quantum computers have the potential to revolutionize various fields, including cryptography (with the ability to break existing encryption schemes), optimization problems, drug discovery, and artificial intelligence. However, it is still early days for the technology, and many challenges need to be overcome before it can be widely deployed.

In summary, quantum computers are an exciting new development with the potential to transform computing and solve problems that are currently considered unsolvable by classical machines. However, they are still in their infancy and require significant research and development before they can be fully realized.
```

非常に精度よく回答できています．この辺りの回答が得意なのは科学技術に関するデータを多く学習できていることによるのかも知れません．

## コード生成(日本語)

```python
messages = []
messages.append({"role": "user", "content": "ロジスティック回帰を学習するためのPythonのコードを生成してください．"})
response = model.chat(tokenizer, messages)
print(response)

# 出力
以下は、Pythonでロジスティック回帰を学習するための簡単なサンプルコードです。このコードは、`sklearn` パッケージを使用します。

# 必要なライブラリのインポート
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# データの読み込み
# ここでは、データフレーム形式の入力データを想定
X = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
]
y = [0, 1, 0, 1]  # 0: クラスA, 1: クラスB

# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ロジスティック回帰モデルの作成
log_reg = LogisticRegression()

# モデルを学習させる
log_reg.fit(X_train, y_train)

# 予測結果を出力
y_pred = log_reg.predict(X_test)
print("予測結果:", y_pred)

# 評価指標の算出
accuracy = accuracy_score(y_test, y_pred)
print("準確率:", accuracy)
print("分類報告:")
print(classification_report(y_test, y_pred))


このコードは、入力データ `X` と目標変数 `y` から学習データとテストデータを作成し、ロジスティック回帰モデルを学習させ、それらの評価指標を算出します。

```

こちらについても非常にうまく生成できています．コード生成に関してはGPT-4レベルで行えている気がします．

## 計算

```python
messages = []
messages.append({"role": "user", "content": "4367+3291="})
response = model.chat(tokenizer, messages)
print(response)

# 出力
4367 + 3291 = 7658

messages = []
messages.append({"role": "user", "content": "310×19="})
response = model.chat(tokenizer, messages)
print(response)

# 出力
310乘以19等于5890。
```

足し算はかなり大きい桁まで，掛け算は3桁×3桁ぐらいまでなら計算はできました．

# 3. おわりに
今回はBaichuan2の紹介でした．中国語がメインのLLMですが日本語でも十分に精度が高い出力を得られることができました．一部，漢字については日本にない漢字を使って出力されるケースがありましたが問題なく出力された文は読むことができます．コード生成については非常に出力の精度が高く，学習データで多く使っている影響が出ていると思われます．
記事に誤り等ありましたらご指摘いただけますと幸いです。

# 4. 参考文献
- https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat
