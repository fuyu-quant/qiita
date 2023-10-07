---
title: Tree of Thoughts
tags:
  - LLM
  - GPT-4
  - TreeofThoughts
private: false
updated_at: '2023-10-06T02:14:46+09:00'
id: c329d3fa80bda5efd32a
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

こんにちは、[fuyu-quant](https://twitter.com/fuyu_quant)です．

今回はTree of Thoughtsの紹介になります．Tree of ThoughtsはLLMに直接入力しても解くことが難しい探索や先読みが必要なタスクを解く精度を上げるLLMを利用するフレームワークになります．
記事に誤り等ありましたらご指摘いただけますと幸いです。

https://github.com/princeton-nlp/tree-of-thought-llm


# 目次
- [1. Tree of Thought](#1-tree-of-thought)
- [2. 使い方](#2-使い方)
    - [通常のGPT-4で解く](#通常のGPT-4で解く)
    - [Tree of Thoughts](#tree-of-thoughts)
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)


# 1. Tree of Thought

ライセンス:MIT(商用利用可能)
リポジトリ:https://github.com/princeton-nlp/tree-of-thought-llm
論文:https://arxiv.org/abs/2305.10601

Language modelではさまざまなタスクを解くことができるが，探索や先読み，初期の設定が重要な役割を果たすタスクなどを解くことが難しいという課題があります．ToT(Tree of Thought)ではさまざまな推論の経路を考え，選択肢を自己評価し，意思決定を行うことができます．これによりいくつかのタスクにおいての性能を向上させました．


![Tree of Thought.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/0c483b20-3a87-9714-668c-0d36b9f6a509.png)


# 2. 使い方

すぐに試したい方はData Science WikiのページまたはColabのリンクから実行してみてください

<a href="https://www.data-science-wiki.net/article?path=/nlp/llm_framework/tree_of_thoughts.html">
<img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/nlp/llm_framework/tree_of_thought.ipynb)

今回は4つの整数(4,5,6,10)と四則演算を使い24という数字を作ることができるか?という問題設定でTree of Thoughtsを試してみます．

※現状だとこの数字の問題を解くようにアルゴリズムが実装されているみたいで，他の問題をこのOSSを使って解くことは出来なさそうです．しかし，実装は公開されているのでこのアイデアをもとに他の問題を解くための実装はできそうです．

## 通常のGPT-4で解く

まず初めにGPT-4だけで問題が解けないのかを確認します．

```python
!pip install langchain
!pip install openai
```

以下ではLangChain経由でGPT-4を使っています．

```python
from langchain.llms import OpenAI

model_name="gpt-4"
llm = OpenAI(temperature=0, model_name=model_name)
```

GPT-4に直接入力して答えが出力されるか確認してみます．他にはプロンプトの工夫があるかもしれませんが一旦は以下のような文章を入力しています．

```python
prompt = "Can you give me a formula that uses all 4,5,6,10 and only four arithmetic operations and the answer is 24?"

output = llm(prompt)
print(output)

# 出力
Sure, here is a formula:
(10 - 4) * 6 - 5 = 24
```

それでは実際にTree of Thoughtsを使って解いてみましょう．

## Tree of Thoughts

以下のコマンドでインストールする．

```python
!pip install tree-of-thoughts-llm
```

```python
import argparse

from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
```

以下で各種設定を行います．

```python
args = argparse.Namespace(backend='gpt-4', 
                          temperature=0.7,
                          n_generate_sample=1,
                          n_evaluate_sample=3,
                          n_select_sample=5,
                          method_generate='propose',
                          method_evaluate='value',
                          method_select='greedy'
                          )
```

以下のコードを実行するとTree of Thoughtsが実行され，正しい答えとなる式が見つかるまで探索を行います，(結構，API費がかかります．)

```python
task = Game24Task()

# Tree of Thoughts
ys, infos = solve(args, task, 900)
```

以下が出力結果になります．正しい答えを導くことができています．

```python
print(ys[0])

# 出力
10 - 4 = 6 (left: 5 6 6)
5 * 6 = 30 (left: 6 30)
30 - 6 = 24 (left: 24)
Answer: (5 * (10 - 4)) - 6 = 24
```

以下のサイトにはTree of Thoughtsのコードに関する情報が記載されています．実装の詳細が気になった方はぜひ参考にしてみてください．

https://blog.devgenius.io/tree-of-thoughts-implementation-to-solve-the-game-of-24-13fb1fa2034b

# 3. おわりに
今回はTree of Thoughtsの紹介でした．LLMが解くことが難しい問題を探索を行わせることで解かせるのはとても面白しアプローチだと思いました．一方で論文で検証されている使い方などはLLMを使わなくてもより効率的に解く手法が存在しており，Tree of Thoughtsをわざわざ使っている感じがしました．Tree of Thoughtsを積極的に使っていきたい利用シーンなどがあるといいですね.

記事に誤り等ありましたらご指摘いただけますと幸いです。

# 4. 参考文献

- https://github.com/princeton-nlp/tree-of-thought-llm
- https://qiita.com/PND/items/56c5e30d5a568188eb1c
- https://blog.devgenius.io/tree-of-thoughts-implementation-to-solve-the-game-of-24-13fb1fa2034b
