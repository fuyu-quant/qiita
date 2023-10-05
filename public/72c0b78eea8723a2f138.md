---
title: Mistral 7B
tags:
  - LLM
  - Mistral
private: false
updated_at: '2023-10-06T02:08:21+09:00'
id: 72c0b78eea8723a2f138
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

今回はMistral 7Bの紹介になります．7Bモデルにも関わらずLlama2 13BやLlama1 34Bなどのパラメータ数が大きいモデルを超える性能を発揮しています．日本語出力に関して少しだけ違和感がある点もありますが，英語やコード生成，計算などは非常に高い精度で出力することができています．

https://mistral.ai/news/announcing-mistral-7b/


# 目次
- [1. Mistral 7B](#1-mistral-7b)
- [2. 使い方](#2-使い方)
    - [日本語での質問](#日本語での質問)
    - [英語での質問](#英語での質問)
    - [コード生成(日本語)](#コード生成(日本語))
    - [計算](#計算)
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)


# 1. Mistral 7B

ライセンス:Apache 2.0
ホームページ:https://mistral.ai/news/announcing-mistral-7b/
Hugging Face:https://huggingface.co/mistralai

- [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

Mistral 7Bは7.3Bのパラメータを持ち、全てのベンチマークにおいてLlama 2 13Bを、多くのベンチマークにおいてLlama 1 34Bを超える強力なモデルです。また、CodeLlama 7Bに匹敵するコード生成能力を持ちながら、英語のタスクでも高い性能を保持しています。

# 2. 使い方

すぐに試したい方はData Science WikiのページまたはColabのリンクから実行してみてください

<a href="https://www.data-science-wiki.net/article?path=/nlp/text_generation(japanese)/mistral_7b_instruct_v01.html">
<img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/nlp/text_generation(japanese)/mistral_7b_instruct_v01.ipyn.ipynb)

今回はMistral-7B-Instruct-v0.1を試していきたいと思います．GPUのメモリは29.2GBを消費しました。

必要なライブラリをインストールします．

```python
!pip install transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

モデルのダウンロード

```python
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
```

## ****日本語での質問****

```python
prompt = """
量子コンピュータについて教えてください。
"""

messages = [
    {"role": "user", "content": f"{prompt}"},
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

# 出力
<s>[INST] 
量子コンピュータについて教えてください。
 [/INST] 量子コンピュータ（Quantum Computers）は、量子力学の理念を基にデジタルデータを处理します。量子力学は、光子、エリトン、磁場などを採用して課題を解決することであり、自然の現象を解決するための数学の方法を示しています。

量子コンピュータは、量子ビットの基本 unit として使われ、量子ビットは2状態を持ったものです。この量子ビットは、自然数0と1を示し、またはスクラムプブリルの形で、0と1の状態の混合を表現します。一般的に、量子コンピュータでは、量子ビットの状態を使って転移を指定するようにデータは表現されます。

量子ビットの重大な特徵は、SuperpositionとEntanglementの2つです。Superpositionは、量子ビットが同時に複数の状態を持つことを示します。例えば、量子ビットは、0 AND 1、0 OR 1の3つの状態のすべてを同時に持つことができます。Entanglementは、量子ビットの緊密な相関についてする詝語です。量子ビットが別の量子ビットと緊密に相関されると、そのquantum stateは、そのための量子ビットが持つ状態のみであればなくないことになります.

量子コンピュータでは、量子ビットを使ってQuantum Gateを作成することができます。Quantum Gateは、量子ビットから量子ビットへの指定した転移を作成し、データを处理するための基本的な操作として使われます。量子コンピュータのアルゴリズムでは、量子ゲートを使って、問題を解決するかどうかを決定します。

量子コンピュータは、特定の問題を解決するために使われ、そのためのアルゴリズムは、自然数の範側において数百、数千以上の量子ビットを使った量子コンピュータに関する問題を解決するデータを処理できます。量子コンピュータによって解決する問題の種類は、現在には有限ですが、実際には量子コンピュータは各種の問題を解決することができることを示しています。量子コンピュータは、大統一的で計算的な計算方法の中であるため、現在からすぐには、量子コンピュータで解決する問題の種類は墙が立っていませんが、未来には量子コンピュータは一緒に、それらの問題を解決することができるようになると思われます。</s>
```



## 英語での質問

```python
prompt = """
Please tell me about quantum computers.
"""

messages = [
    {"role": "user", "content": f"{prompt}"},
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

# 出力
<s>[INST] 
Please tell me about quantum computers.
 [/INST] Quantum computers are a type of computer that use the principles of quantum mechanics to store, process, and manipulate information. Unlike traditional computers, which use bits represented as either 0 or 1, quantum computers use quantum bits, or qubits, which can exist in a superposition of states, meaning they can represent 0, 1, or a combination of both at the same time.

This ability to represent multiple states simultaneously allows quantum computers to perform certain calculations much faster than traditional computers. While traditional computers are limited by the binary nature of classical computing, quantum computers use quantum mechanics to manipulate and process information in a whole new way, offering the potential for exponential speedups in certain types of calculations.

Quantum mechanics is a branch of physics that describes the behavior of matter and energy at the atomic and subatomic level. It is a complex and abstract mathematical theory, but in simple terms, it allows us to understand the behavior of particles such as electrons and photons, which are the fundamental building blocks of matter and energy.

One of the most promising applications of quantum computers is in the field of cryptography. Quantum computers are capable of breaking many of the cryptographic algorithms currently used to secure online communication. However, they also have the potential to create new, unbreakable cryptographic algorithms, which could revolutionize the way we secure online data and communication.

Another area where quantum computers are expected to have a significant impact is in materials science and chemistry. Quantum mechanics plays a key role in understanding the behavior of atoms and molecules, and quantum computers could be used to simulate the behavior of complex materials and chemical systems in ways that traditional computers are not suited for.

Overall, quantum computers offer the potential for exponential speedups in certain calculations, and their ability to represent multiple states simultaneously could revolutionize the way we think about computing. While they are still a relatively new technology, they are likely to play an increasingly important role in the development of new technologies and applications in the coming years.</s
```

## ****コード生成(日本語)****

```python
prompt = """
LightGBMを学習するためのPythonのコードを生成してください．
"""

messages = [
    {"role": "user", "content": f"{prompt}"},
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

# 出力
<s>[INST] 
LightGBMを学習するためのPythonのコードを生成してください．
 [/INST] 以下のPythonコードを使用してLightGBMを学習することができます。LightGBMの学習はデータセットについて必要ないといったものではないため、データセットを指定するようにしてください。


from sklearn import datasets
import numpy as np
import pandas as pd

# データセットを指定する必要がありません
# datasets.load_breast_cancer()
# X, y = datasets.load_iris()
# X, y = datasets.load_digits()
# X, y = datasets.load_wine()

# データсеットを加工して使用する必要があります
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv', int64=True)

# データセットを加工します
# Pandasでデータをデータフレームで格納していますが、LightGBMはNumpyのアレイなどを使っています
X = np.array(X).reshape(-1, X.shape[1])

# LightGBMに入力などを準備します
params = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 5
}

model = LightGBM.fit(X, y, **params)


LightGBMは使いやすいですが、このLightGBMの学習を使って何を実現できますようになりますか？</s>
```

最後に質問で聞き返されている箇所は気になりますが，コード生成もかなり精度良く出力を得られています．

## 計算

```python
prompt = """
6712+3957=
"""

messages = [
    {"role": "user", "content": f"{prompt}"},
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])

<s>[INST] 
6712+3957=
 [/INST] 10679</s>
```

計算問題についてはBaichuan2やXwin-LMなどのモデルを超える精度でうまく計算できました．

# 3. おわりに

今回はMistral 7Bの紹介でした．Llama2の13Bモデルを全てのベンチマークで勝ち，Llama1 34Bモデルにいくつかのベンチマークで勝利しています．モデルの小規模高性能化の研究が進んでいることを感じました．日本語の性能があまり良くないのは少し学習データが少ないせいかもしれません．

# 4. 参考文献
* https://mistral.ai/news/announcing-mistral-7b/
