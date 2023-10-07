---
title: CreateToolAGI：ChatGPTとLangChainで何でもできるAI
tags:
  - OpenAI
  - ChatGPT
  - langchain
  - 記事投稿キャンペーン_ChatGPT
  - GPT-4
private: false
updated_at: '2023-06-12T17:42:43+09:00'
id: 5e5b4f429e775ac39a74
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに
こんにちは、[fuyu-quant](https://twitter.com/fuyu_quant)です!
今回はLangChainのToolsという機能とChatGPTを使い，CreateToolAGIというものを作成しました．

　CreateToolAGIは，**LLMが問題を受け取った時にその問題を一般化し，一般化した問題を解くための道具(pythonスクリプト)を作り，その道具を使いもとの問題を解くことをできるようにするLLMの拡張機能**です．**さらに作成した道具は使いまわすことができるため，できることを増やし能力を拡張していけます**．

(※この記事の「ChatGPT」は「ChatGPT API」の意味になります)

https://github.com/fuyu-quant/CreateTool-AGI

![createtoolagi.gif](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/1bbbddfc-0f5e-61e8-835d-510d5dc2e2d9.gif)
(※動画では再生速度をはやくしています)

# 目次
- [1. CreateToolAGI](#1-createtoolagi)
- [2. 使い方](#2-使い方)
- [3. 実行例](#3-実行例)
- [4. 今後の展望](#4-今後の展望)
- [5. 参考文献](#5-参考文献)

# 1. CreateToolAGI
## LangChainのToolsとは

　LangChainのToolsとはLLMにさまざまな機能を持たせるためのものです．以下のリンク先に実行例や説明などがあるので詳細について知りたい方は確認してみてください．

https://qiita.com/fuyu_quant/items/c0b29f037b7834c19e2c#langchainのtoolとは

以下は公式のリンクです．

https://python.langchain.com/en/latest/modules/agents/tools.html

## CreateToolAGIとは

- CreateToolAGIのコンセプト
LLMが直接は実行できないことも人間のように道具を作り，それを使うことで解決できるようにすることを目標に作成しました．(現在は英語のみの対応になっています)
また以下のような工夫点があります．
    - 入力された問題の一般化
    汎用的に使える道具を作るため
    - 道具を作る必要があるかどうかの判定
    - 作成した道具を外部ファイルに書き出し保存する．
    同じような入力に対しては道具を使い回して利用できるようにするため．
    - 作成した道具が使えなかった場合に，作成した道具をもとに新たに道具を作り直す

- CreateToolAGI内の処理
内部の処理は以下のようになっています．  
    - Step1：入力として受け取った問題の一般化を行う．
    - Step2：一般化した問題がすでに所持しているToolsで実行できるかを判定する
        - できる場合：Toolsを使い最終出力を得る．
        - できない場合：一般化した問題を解くためのToolsを作成する．
            - この時にToolsが正しく使えない場合は，一度作ったものを参考にして再度作り直す
            - 作成したToolsを使い最終出力を得る．
            
    
処理の全体像は以下のようになっています．
![createtoolagi.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/7dfa0258-5c90-a0d8-54af-a512b81dc4c3.png)


# 2. 使い方

CreateToolAGIの使い方の紹介です．

以下のノートブックでも実行を確認できます．
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/CreateTool-AGI/blob/main/examples/createtoolagi.ipynb)
はじめに必要なライブラリをinstallします．

```python
# CreateToolAGIのinstall
!pip install git+https://github.com/fuyu-quant/CreateTool-AGI.git

!pip install langchain==0.0.167
!pip install openai==0.27.4
```

YOUR API KEYの箇所は[OpeaAIのAPI key](https://platform.openai.com/account/api-keys)を取得し，書き換えてください．
現在は，GPT-4での実行にのみ対応しているのでGPT-4が使えるAPI keyをご利用ください．

```python
from createtoolagi import CreateToolAGI

import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
```

これで準備は完了しました．

# 3. 実行例

いくつかの実行例を掲載します．(※現在はそこまで複雑な問題を解くことができません)

- 通常の実行
```python
input = 'What is the sum of the prime numbers from 1 to 100000?'
tools = []
ctagi = CreateToolAGI()

ctagi.run(input, tools)
```
実行結果
```python
> Generalize the input task.
Generalized Task：What is the sum of the prime numbers from A to B?

> Determine if you should make a tool.
I must create the tool before executing.

try:1
> Create a tool.
Completed!
Created tool name：PrimeSumTool

> Entering new AgentExecutor chain...
I need to find the sum of prime numbers between 1 and 100000.
Action: PrimeSumTool
Action Input: 1,100000
Observation: 454396537
Thought:I now know the final answer.
Final Answer: 454396537

> Finished chain.
```

- Toolsをファイルに書き出すver
```python
# 質問
input = 'Sum of prime numbers from 1 to 50'
tools = []
# 保存するフォルダーの指定
folder_path = '/content/'

ctagi = CreateToolAGI()

ctagi.run(input, tools, folder_path)
```

実行結果
```python
> Generalize the input task.
Generalized Task：Sum of prime numbers from A to B

> Determine if you should make a tool.
I must create the tool before executing.

Try:1
> Create a tool.
Completed!
Created tool name：PrimeSumTool

> Entering new AgentExecutor chain...
Error occurred: Could not parse LLM output: `I need to find the sum of prime numbers between 1 and 50. I can use the PrimeSumTool for this.`

Try:2
> Create a tool.
Completed!
Created tool name：PrimeSumTool

> Entering new AgentExecutor chain...
I need to use the PrimeSumTool to find the sum of prime numbers between 1 and 50.
Action: PrimeSumTool
Action Input: 1,50
Observation: 328
Thought:I now know the final answer.
Final Answer: 328

> Finished chain
```

- 作成したToolsの利用

```
# 作成したToolsの名前によって異なる
from PrimeSumTool import PrimeSumTool

input = 'Sum of prime numbers from 51 to 100'

# Use the tool you created
tools = [PrimeSumTool()]

ctagi = CreateToolAGI()

ctagi.run(input, tools)
```

出力結果

```python
> Generalize the input task.
Generalized Task：Sum of prime numbers from A to B

> Determine if you should make a tool.
I do not need to create a tool to run it.

> Entering new AgentExecutor chain...
I need to find the sum of prime numbers between 51 and 100.
Action: PrimeSumTool
Action Input: 51,100
Observation: 732
Thought:I now know the final answer.
Final Answer: 732

> Finished chain.
```

　上記のような問題設定であればGPT-4などで正しい回答を得られます．しかし上記のような問題でも”What is the sum of the prime numbers from 1 to 100000000?”などの多くの計算量を伴う場合は実行できませんが，CreateToolAGIではpythonスクリプトを生成しているので理論上は実行できます．

今後は，機械学習アルゴリズムなども生成できるようにしたいと考えています．



# 4. 今後の展望
最後までお読みいただきありがとうございます．今回はCreateToolAGIの紹介でした．

今後はCreateToolAGIがより使いやすくなるようにアップデートしていければと考えています．また研究的な応用や社会実装などでの使い道なども考えていきたいです．

また，現在ではまだいくつかの課題があります．
- 複雑な処理の場合にうまくToolsを作ることができない．
    - 似ている処理の例をIn-context learningにより学習されば実行できる可能性がある．
    - [BabyAGI](https://github.com/yoheinakajima/babyagi)などと連携させることでタスクを分解し対応できるようにする．
- CreateToolAGIにできてGPT-4にできない問題の探索
    - 「実行例」の最後にも記述した，計算量が多くなる場合はCreateToolAGIを使わないと直接は求められないと考えています．
    - 機械学習アルゴリズムの実行などもできるようにする．
- 作成した大量のToolsへのアクセス方法

などなど...
他にもいろいろありますが，より使いやすくなるように更新していきたいです．

今回の記事の内容で

# 5. 参考文献

- [https://qiita.com/fuyu_quant/items/c0b29f037b7834c19e2c](https://qiita.com/fuyu_quant/items/c0b29f037b7834c19e2c)
- [https://github.com/yoheinakajima/babyagi](https://github.com/yoheinakajima/babyagi)
- [https://github.com/Significant-Gravitas/Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT)
