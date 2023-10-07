---
title: Automatic Prompt Engineer
tags:
  - OSS
  - OpenAI
  - LLM
  - GPT-4
private: false
updated_at: '2023-10-07T18:59:14+09:00'
id: d90c7d69b143d59a7b84
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

今回はAutomatic Prompt Engineer(APE)の紹介になります．Automatic Prompt Engineerでは入力と出力のペアを与えることで，その入力から出力を得るためのプロンプトを自動で生成することができます．さらにどのようなプロンプトが良いかや人間が作成したプロンプトの評価なども行うことができます．これによりいくつかのデータセットでIn-context LearningやChain-of-thoughtの性能を上げるような結果が出しています．

記事に誤り等ありましたらご指摘いただけますと幸いです。

https://github.com/keirp/automatic_prompt_engineer

# 目次
- [1. Automatic Prompt Engineer](#1-automatic-prompt-engineer)
- [2. 使い方](#2.-使い方)
  - [プロンプトの探索](#プロンプトの探索)
  - [人が作成したプロンプトの評価](#人が作成したプロンプトの評価)
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)


# 1. Automatic Prompt Engineer

ライセンス:MIT

リポジトリ:https://github.com/keirp/automatic_prompt_engineer
公式サイト:https://sites.google.com/view/automatic-prompt-engineer
論文:https://arxiv.org/abs/2211.01910

Automatic Prompt Engineerでは入力と出力のペアと，プロンプトのテンプレートを入力として与えることで，入力が与えられた時に正確に出力が得られるようにプロンプトテンプレートを最適化することができます．以下が内部での主な処理になります．詳細については論文を参照してください．

![ape.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/00c5d5b9-0547-c269-6913-903e63fcac1d.png)


# 2. 使い方

すぐに試したい方はData Science WikiのページまたはColabのリンクから実行してみてください

<a href="https://www.data-science-wiki.net/article?path=/nlp/llm_framework/automatic_prompt_engineer.html">
<img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>

<a href="https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/develop/nlp/llm_framework/automatic_prompt_engineer.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

今回は公式の紹介しているデモを紹介していこうと思います．内容としては入力した言葉と反対の言葉が出力されるようにするプロンプトの最適化を行います．

以下のコマンドでインストールします．

```python
!pip install git+https://github.com/keirp/automatic_prompt_engineer
```

```python
from automatic_prompt_engineer import ape

import openai
openai.api_key = ''
```

## プロンプトの探索

以下では入力するデータと出力として得たいデータのセットを用意しています．

```python
words = ["sane", "direct", "informally", "unpopular", "subtractive", "nonresidential", "inexact", "uptown", "incomparable", "powerful", "gaseous", "evenly", "formality", "deliberately", "off"]
    
antonyms = ["insane", "indirect", "formally", "popular", "additive", "residential", "exact", "downtown", "comparable", "powerless", "solid", "unevenly", "informality", "accidentally", "on"]
```

以下が評価を行うためのテンプレートになります．

```python
eval_template = \
"""Instruction: [PROMPT]
Input: [INPUT]
Output: [OUTPUT]"""
```

以下のコマンドで質の高いプロンプトを生成し，最もよいプロンプトを見つけます．

```python
result, demo_fn = ape.simple_ape(
    # 入出力のペア
    dataset=(words, antonyms),
    # プロンプトのテンプレート
    eval_template=eval_template,
)
```

出力結果になります．scoreの高い値が今回のデータセットにおいて最もInputを入力した時にOutputを得やすいプロンプトです．scoreの値はInstructionとInputが与えられた時の得たいOutputを出力する確率の対数をとったものになります．

```python
print(result)

# 出力
score: prompt
----------------
-0.17:  write the input word with the opposite meaning.
-0.21:  write down the opposite of the word given.
-0.25:  find the antonym (opposite) of each word.
-0.28:  produce an antonym (opposite) for each word given.
-0.42:  make a list of antonyms.
-0.44:  "list antonyms for the following words".
-0.76:  produce an output that is the opposite of the input.
-5.44:  "Add the prefix 'in' to each word."
-5.79:  reverse the order of the input.
-7.37:  reverse the order of the letters in each word.
```

## 人が作成したプロンプトの評価

```python
# 評価するプロンプト
manual_prompt = "Write an antonym to the following word."

human_result = ape.simple_eval(
    dataset=(words, antonyms),
    eval_template=eval_template,
    prompts=[manual_prompt],
)
```

```python
print(human_result)

# 出力
log(p): prompt
----------------
-0.24: Write an antonym to the following word.
```

# 4. おわりに

今回はAutomatic Prompt Engineerの紹介でした．社会実装の場面においても割と使いやすく応用範囲が広いようなOSSだと感じました．OpenAI以外のモデルでも使えるようになると活用が広がりそうです．LLMのプロンプトエンジニアリングに関する研究も進んできていますね．

記事に誤り等ありましたらご指摘いただけますと幸いです。

# 5. 参考文献

- [https://sites.google.com/view/automatic-prompt-engineer](https://sites.google.com/view/automatic-prompt-engineer)
