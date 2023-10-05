---
title: Qwen
tags:
  - LLM
private: false
updated_at: '2023-10-06T02:07:30+09:00'
id: c6e6d1eaaa9e76a453af
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

今回はQwenの紹介になります．アリババクラウドが開発したモデルでLlama2の2兆トークンを超える3兆トークンのデータセットで事前学習を行なったモデルで14Bのモデルでは一部の評価指標ではLlama2-70Bに迫る精度を出しているそうです．
日本語，英語，コード生成，計算などの出力を試してみたいと思います．

https://github.com/QwenLM/Qwen/tree/main

# 目次
- [1. Qwen](#1-qwen)
- [2. 使い方](#2-使い方)
    - [日本語での質問](#日本語での質問)
    - [英語での質問](#英語での質問)
    - [コード生成(日本語)](#コード生成(日本語))
    - [計算](#計算)
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)

# 1. Qwen

ライセンス:[LICENSE](https://github.com/QwenLM/Qwen/blob/main/LICENSE)
商用利用したい場合は問い合わせが必要みたいです.  
https://dashscope.console.aliyun.com/openModelApply/qianwen
リポジトリ:https://github.com/QwenLM/Qwen/blob/main/README_JA.md
Hugging Face:https://huggingface.co/Qwen/Qwen-14B-Chat-Int4

アリババクラウドが開発した英語と中国語を中心に様々なドメインについて3兆トークンの多言語データセットで学習を行なったモデルです．モデルは7Bと14Bのモデルの通常のモデル，チャットモデル，量子化したチャットモデルの6種類あります．

- [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)
- [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
- [Qwen-7B-Chat-Int4](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4)
- [Qwen-14B](https://huggingface.co/Qwen/Qwen-14B)
- [Qwen-14B-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)
- [Qwen-14B-Chat-Int4](https://huggingface.co/Qwen/Qwen-14B-Chat-Int4)

# 2. 使い方
すぐに試したい方はData Science WikiのページまたはColabのリンクから実行してみてください

<a href="https://www.data-science-wiki.net/article?path=/nlp/text_generation(japanese)/qwen_14b_chat_int4.html">
<img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/nlp/text_generation(japanese)/qwen_14b_chat_int4.ipynb)

今回はQwen-14Bに対してSFTとRLHFを行い作成したチャットモデルをInt4の量子化したモデル[Qwen/Qwen-14B-Chat-Int4](https://huggingface.co/Qwen/Qwen-14B-Chat-Int4)を使います．

必要なライブラリをインストールします．

```python
!pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
!pip install auto-gptq optimum
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

モデルのダウンロード

```python
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-14B-Chat-Int4",
    trust_remote_code=True
    )

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-14B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
    ).eval()
```

## 日本語での質問

```python
response, history = model.chat(
    tokenizer,
    "量子コンピュータについて教えてください．",
    history=None
    )
print(response)

# 出力
量子コンピュータは、通常のコンピュータとは異なるアルゴリズムと原理を使用して、非常に高速に計算を行うことができます。量子コンピュータは、量子力学に基づく概念を使用して、量子ビット（qubits）という最小単位の情報を受け取り、処理します。これらのqubitsは、通常のコンピュータのビットとは異なり、0と1の状態だけでなく、両方の状態を同時に保持することができます。このため、量子コンピュータは、複雑な問題を非常に高速に解決することができます。
```


## 英語での質問

```python
response, history = model.chat(
    tokenizer, 
    "Please tell me about quantum computers.",
    history=None
    )
print(response)

# 出力
Quantum computers are a type of computer that use quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data. Unlike classical computers, which use bits to represent information, quantum computers use quantum bits, or qubits. 

One of the key features of quantum computers is their ability to perform certain calculations much faster than classical computers. This is because quantum computers can perform many calculations simultaneously, thanks to the phenomenon of superposition. In a superposition, a qubit can exist in multiple states at the same time, allowing it to perform multiple calculations simultaneously.

Another important feature of quantum computers is their ability to perform certain types of calculations that are impossible for classical computers. This is because quantum computers can take advantage of the phenomenon of entanglement, which allows qubits to be connected in a way that allows them to affect each other even when they are separated by large distances.

Quantum computers are still in the early stages of development, and there are many technical challenges that need to be overcome before they can be widely used. However, they have the potential to revolutionize many fields, including cryptography, chemistry, and materials science.
```

## コード生成(日本語)

```python
response, history = model.chat(
    tokenizer, 
    "LightGBMを学習するためのPythonのコードを生成してください．",
    history=None
    )
print(response)

# 出力
以下は、LightGBMを使用してPythonで学習するためのコード例です。
```python
import lightgbm as lgb
import pandas as pd

# データの読み込み
data = pd.read_csv('data.csv')

# データの前処処理
# ... (データの前処処理のコード)

# データをLightGBM用のデータ形式に変換
lgb_data = lgb.Dataset(data['feature'], label=data['target'])

# モデルの学習
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
model = lgb.train(params, lgb_data, num_boost_round=100)

# モデルの予測
y_pred = model.predict(data['feature'])

# 予測結果の出力
print(y_pred)

このコードでは、LightGBMを使用して、CSVファイルから読み込んだデータを学習し、予測を行っています。データの前処処理やモデルのパラメータなどの詳細は、プロジェクトによって異なります。
```

コード生成もかなり精度良く出力を得られています．

## 計算

```python
response, history = model.chat(
    tokenizer,
    "6712+3957=",
    history=None
    )
print(response)

# 出力
10669

# 掛け算
response, history = model.chat(
    tokenizer,
    "12×38=",
    history=None
    )
print(response)

# 出力
456
```

計算問題についてはBaichuan2やXwin-LMなどのモデルを超える精度でうまく計算できました．

# 3. おわりに
Qwenの紹介でした．日本語，英語，コード生成，計算問題ともに違和感なく出力を得られています．計算問題ではBaichuan2やXwin-LMでは上手くいかないことが多かったですが、Qwen-14B-Chat-Int4では上手く実行できています。GPT-3.5-turboを超える精度を達成する日も近いかもしれません．

# 4. 参考文献
- https://36kr.jp/253527/
