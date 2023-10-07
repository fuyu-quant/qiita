---
title: ChatGPTとLangChainで何でもできるAIを作る
tags:
  - OpenAI
  - gpt-3
  - ChatGPT
  - langchain
  - 記事投稿キャンペーン_ChatGPT
private: false
updated_at: '2023-06-19T16:22:11+09:00'
id: c0b29f037b7834c19e2c
organization_url_name: brainpad
slide: false
ignorePublish: false
---
　この記事は[記事投稿キャンペーン_ChatGPT](https://qiita.com/official-events/b80ebe828fa453edf260)の記事です。

以下は、個人開発した最新のものになります．
[CreateToolAGI：ChatGPTとLangChainで何でもできるAI](https://qiita.com/fuyu_quant/items/5e5b4f429e775ac39a74)

# はじめに
こんにちは、[fuyu-quant](https://twitter.com/fuyu_quant)です．
　今回はLangChainという「大規模言語モデルを使いこなすためのライブラリ」の紹介とその機能を発展させるために作った新しいライブラリ[langchain-tools](https://github.com/fuyu-quant/langchain-tools)の説明およびその可能性について共有したいと思います．
　LangChainの機能であるtoolを使うことで，**プログラムとして実装できるほぼ全てのことがChatGPTなどのモデルで自然言語により実行できる**ようになります．今回は自然言語での入力により機械学習モデル(LightGBM)の学習および推論を行う方法を紹介します．

記事に誤り等ありましたらご指摘いただけますと幸いです。

(※この記事の「ChatGPT」は「ChatGPT API」の意味になります)

# 目次
- [1. LangChainとは](#1-LangChainとは)
- [2. langchain-toolsについて](#2-langchain-toolsについて)
- [3. langchain-toolsの活用例](#3-langchain-toolsの活用例)
- [4. LangChainのtoolとlangchain-toolsの可能性](#4-langChainのtoolとlangchain-toolsの可能性)
- [5. おわりに](#5-おわりに)
- [6. 参考文献](#6-参考文献)


# 1. LangChainとは
　[LangChain](https://python.langchain.com/docs/get_started/introduction.html)とは，ものすごく簡単にいうと**大規模言語モデルにさまざまな機能を付け加えるのに便利なもの**です(超個人的な意見です)
[公式](https://github.com/hwchase17/langchain#-what-is-this)では以下のように紹介されています．

「大規模言語モデル（LLM）は、開発者がこれまでできなかったようなアプリケーションを構築できるようにする、革新的なテクノロジーとして台頭しています。しかし、LLMを単独で使用しても、真にパワフルなアプリケーションを作るには不十分な場合が多く、LLMを他の計算ソースや知識と組み合わせることができたときに、真のパワーが発揮されます。
このライブラリは、このようなタイプのアプリケーションの開発を支援することを目的としています。」

と，書かれています．わりと新めのツールですがさまざまな機能があります．
以下にLangChainのさまざまな実行例を載せているので参考にしてみてください．

https://github.com/fuyu-quant/langchain

### LangChainのtoolとは
　LangChainにはさまざまな機能があります．その中でも大規模言語モデルにさまざまな能力を持たせるための**tool**というものがあります．このtoolが非常に便利です．
　それではこのtoolを使う場合と使わない場合について見てみましょう．

* **ツールを使わない場合**
```python
message = [HumanMessage(content='株式会社ブレインパッドの創業から100年後は何年になる?')]
llm(message)
```
実行結果
```
AIMessage(content='株式会社ブレインパッドは、2021年に創業したため、100年後は2121年になります。')
```
よく分からない結果になってしまいました．これは「株式会社ブレインパッド」に関する知識がLLMにないためです．

* **ツールを使う場合**
ここでLangChainのtoolにより，以下のように検索能力("serpapi")と計算能力("llm-math")を与えこれを実行すると...
```python
tools = load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0))

response = agent({"input":"株式会社ブレインパッドの創業から100年ごは何年になる?"})
```
実行結果
```python
> Entering new AgentExecutor chain...
 I need to figure out when Brainpad was founded.
Action: Search
Action Input: "株式会社ブレインパッド 創業"
Observation: March 18, 2004
Thought: I need to calculate how many years have passed since then.
Action: Calculator
Action Input: 2004 + 100
Observation: Answer: 2104
Thought:WARNING:root:Failed to persist run: HTTPConnectionPool(host='localhost', port=8000): Max retries exceeded with url: /chain-runs (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0xffff6651f9a0>: Failed to establish a new connection: [Errno 111] Connection refused'))
 I now know the final answer.
Final Answer: 2104年
```
上記のように検索を行い情報を抽出し，さらにその結果をもとに計算をおこない正しい結果を返してくれます．
(この実行方法の詳細について[ノートブック](https://github.com/fuyu-quant/langchain/blob/main/examples/search_chat_model.ipynb)を参考にしてください)

これらの"serpapi"(googleの検索API)や"llm-math"などはデフォルトで提供されているものになります．toolでは他にもさまざまなAPIによるtoolを使うことができます．しかし，公式が用意しているもの以外にも新たなtoolを開発してみたいですよね？

**便利なことにLangChainにはこのtoolを作るための方法があります.しかもpythonの関数の形で定義するだけでtool化できます!!**
ここの[リンク](https://python.langchain.com/docs/modules/agents/tools/how_to/custom_tools)が参考になると思います．

# 2. langchain-toolsについて
　大規模言語モデルにさまざまな機能を追加するにはさまざまな**tool**を用意すればよいことが分かりました．今回は新たに作成したLangChainのtoolのサンプルを紹介します．また，実際にさまざまなツールをLLMに持たせるためにはインプットやアウトプットの形式，内部での処理をある程度統一する必要があると思いました．そのための一つの基準になるようなものとして[langchain-tools](https://github.com/fuyu-quant/langchain-tools)というものを作成しました．
(まだtoolとして用意できているものは少ないですが...)

現在，langchain-toolsでは主に以下の二つの機能を公開しています．
* mltools
    * lightgbm-tools...LightGBMの学習と推論をプロンプトで行えるようにするtool
    
* toolmaker...プロンプトの入力からtoolそのものを作るtool(精度は良くないです)

　使える**tool**を増やしていくことで**GPT-3やGPT-4などのLLMでコードにできることならほとんど何でも実行できるようになる**と考えています。

# 3. langchain-toolsの活用例

### lightgbm-tools

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain-tools/blob/main/examples/langchain-tools_LightGBM.ipynb)
lightgbm-toolsは**プロンプトでLightGBMの学習や推論が必要な入力を受けた際に実行されるtoolで、実際に内部でLightGBMの学習や推論が行われています**.以下のように設定することで実行できます．
```python
tools = [
    Tool(name = 'lgbm_train_tool',func = mltools.lgbm_train_tool, description=""),
    Tool(name = 'lgbm_inference_tool',func = mltools.lgbm_inference_tool, description="")
    ]
```
```python
agent.run("train.csvを使ってLightGBMの学習を行なったあと，test.csvのデータを推論してください")
```
以下のような実行ログとなります．
```python
> Entering new AgentExecutor chain...
 LightGBMを使って学習と推論を行なう
Action: lgbm_train_tool
Action Input: train.csv

Observation: LightGBMの学習が完了しました
推論を行なう
Action: lgbm_inference_tool
Action Input: test.csv

Observation: LightGBMの推論が完了しました
Thought: 結果を確認する
Final Answer: LightGBMの学習と推論が完了しました

> Finished chain.
LightGBMの学習と推論が完了しました
```
上記のような結果になり，実際に学習とその推論結果をcsvファイル形式で出力することができます．


### toolmaker

※以下のColabの実装はあまり精度がよくないので試したい場合は下のリンク先のページから試してみてください．
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain-tools/blob/main/examples/langchain-tools_toolmaker.ipynb)
**※以下のリンクは関連した内容で個人開発したコードとその説明になります．toolmakerに近いものを試す場合は以下のリンクを確認してみてください．**
[CreateToolAGI：ChatGPTとLangChainで何でもできるAI](https://qiita.com/fuyu_quant/items/5e5b4f429e775ac39a74)


　toolmakerは**新しくLLMで使いたい機能を自然言語の入力だけでtool化するためのtool**です．toolmakerによって新たなtoolを作成できるのでよりLLMの機能を拡充していけます．
※現在のtoolmakerの性能では完璧な精度でtool化できていないのが現状です．改善点などがあればぜひ教えていただきたいです．

```python
tools = [Tool(name = 'toolmaker', func = toolmaker.toolmaker, description="")]

tool_name = "XGBoost training"
agent.run(f"Please make a langchain-tool for {tool_name}")
```
出力結果は
```python
> Entering new AgentExecutor chain...
 I need to find a tool that can help me with XGBoost training
Action: toolmaker
Action Input: XGBoost training
Observation: 
@tool('xgboost_training_tool)
def xgboost_training_tool(query: str) -> str:
    'useful for XGBoost training'

    #importing libraries
    import xgboost as xgb
    from xgboost import XGBClassifier

    #instantiating XGBoost classifier
    xgb_clf = XGBClassifier()

    #fitting the model
    xgb_clf.fit(X_train, y_train)

    #predicting on test data
    y_pred = xgb_clf.predict(X_test)

    #calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    result = 'finish xgboost_training
    return result
Thought: I now know the final answer
Final Answer: The final answer is a langchain-tool for XGBoost training that can be used to fit a model, predict on test data, and calculate accuracy.
```
この新たに作成したtoolはそのままでは使えないので改良が必要です...
完璧なtoolを作ることができれば非常に便利なんですが...

# 4. LangChainのtoolとlangchain-toolsの可能性
　LangChainのtoolの機能を使うことで**さまざまなAPIからの情報取得や関数の実行**を行うことができました．
**これは自然言語のみの入力で情報の取得から，機械学習モデルの訓練，学習を行なったモデルによる推論などのさまざまなタスクを文章の入力だけで実行できることを意味しています．**
　langchain-toolsなどでさまざまなtoolを用意できれば、**どんな高度な質問やタスクに対しても回答や実行することができる汎用AIの一つの形**になるかもしれません(大げさ)

また、大規模言語モデルの企業や社会への導入が進むと以下のようなことが重要になってくると考えています．

* **システム内部は秘密にしたいが公開したい各企業のサービスをAPIとして利用できるようにしtool化**
    * **LangChainで使える形で公開することで自然言語により誰でも簡単に使うことができる**
    * **多くの人に使ってもらえるようになるため有料のAPIとして公開することでマネタイズしやすい**
    
* **オーブンソースのコードのtool化**
    * **プログラミングなどをあまり使いたくない人でも自然言語で実行できる**
    * **toolmakerによりtoolを作成する手間もいつかは省けるかも?**

# 5.　おわりに
　長くて分かりにくい説明に最後まで付き合っていただきありがとうございます．
langchain-toolsでは今後もより**便利な機能の実装**，**LLMで呼び出しやすい形式の調整**や**toolの統一化**などに取り組んでいきたいと考えています．
もし開発に興味がある方や改善点などがあればいればいつでもご連絡ください．(開発能力は低めなのでとても助かります...)

# 6. 参考文献
* https://github.com/hwchase17/langchain
* https://python.langchain.com/docs/modules/agents/tools/
* https://python.langchain.com/docs/modules/agents/
* https://note.com/npaka/n/n6b7a07e492f1
* https://qiita.com/fuyu_quant/items/5e5b4f429e775ac39a74
