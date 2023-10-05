---
title: LangChain 20230423 update
tags:
  - ChatGPT
  - langchain
  - GPT-4
private: false
updated_at: '2023-04-24T00:55:17+09:00'
id: af7e5384ec9de78b867e
organization_url_name: null
slide: false
ignorePublish: false
---
ver 0.0.147

# はじめに
LangChainの2023/4/23のupdate情報です．以下のtwitterのリンクを参考にしています．

- ChatGPT Data Loader
- PowerBI Dataset Agent
- MyScale
- AnalyticDB
- Python Document Loader

https://twitter.com/LangChainAI/status/1649815076403281920?s=20

記事に誤り等ありましたらご指摘いただけますと幸いです．

備考：LangChainの実装まとめ
https://github.com/fuyu-quant/langchain

# 目次
- [updateの概要](#updateの概要)
- [ChatGPT Data Loader](#chatgpt-data-loader)
- [PowerBI Dataset Agent](#powerbi-dataset-agent)
- [MyScale](#myscale)
- [AnalyticDB](#analyticdb)
- [Python Document Loader](#python-document-loader)
- [おわりに](#おわりに)

# updateの概要

LangChainには以下にあるように大きく6つのモジュールで構成されています．
太字の箇所が今回アップデートされた箇所になります．

- Models
- Prompts
- Indexes
    - **Data Loader**
    - **Vectorstores**
- Memory
- Chains
- **Agents**

IndexesのData Loaderに「ChatGPT Data Loader」がVectorestoresに「MyScale」と「AnalyticDB」というサービスとの連携するためのものが追加されました．

またAgentsにはPoserBIと対話をするための「PowerBI Dataset Agent」が追加されました．

その他の機能としてPython Document LoaderというPythonファイルのエンコーディングを自動検出する機能が追加されました．

# ChatGPT Data Loader

https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/chatgpt_loader.html

ChatGPTとの会話履歴の文章を取得するためのData Loader

使い方

- [https://chat.openai.com/](https://chat.openai.com/)にアクセスする
- プロフィールの「Settings」を押し，「Export data」を選択
![スクリーンショット 2023-04-23 23.11.17.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/2d9b6067-31ca-8151-4dbe-454849e59923.png)
- 以下の画面に遷移するので「Confirm export」を選択
![スクリーンショット 2023-04-23 23.11.54.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/917b6b82-b38a-055d-95b6-28235d2ce674.png)
- OpenAIに登録しているアカウントにメールが来るので，そのメールのダウンロードボタンを押すと会話履歴のデータを取得できる．

実装

```python
from langchain.document_loaders.chatgpt import ChatGPTLoader

# ダウンロードしたデータまでのパスを指定
loader = ChatGPTLoader(log_file='', num_logs=1)

loader.load()
```

# PowerBI Dataset Agent

https://python.langchain.com/en/latest/modules/agents/toolkits/examples/powerbi.html

PowerBIのデータセットと対話するためのエージェント

```python
from langchain.agents.agent_toolkits import create_pbi_agent
from langchain.agents.agent_toolkits import PowerBIToolkit
from langchain.utilities.powerbi import PowerBIDataset
from langchain.llms.openai import AzureOpenAI
from langchain.agents import AgentExecutor
from azure.identity import DefaultAzureCredential

llm = AzureOpenAI(temperature=0, deployment_name="text-davinci-003", verbose=True)
toolkit = PowerBIToolkit(
    powerbi=PowerBIDataset(None, "<dataset_id>", ['table1', 'table2'], DefaultAzureCredential()), 
    llm=llm
)

agent_executor = create_pbi_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
)

# 以下のような質問をデータセットに対して行える
agent_executor.run("Describe table1")
agent_executor.run("How many records are in table1?")
```

# MyScale

https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/myscale.html

[MyScale](https://myscale.com/)というベクトル検索とSQLを使った分析が使えるクラウド型データベースと連携するためのもの

# ****AnalyticDB****

https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/analyticdb.html

AnalyticDBと連携するための機能

「AnalyticDBの説明」

AnalyticDB for PostgreSQL は、オープンソースの Greenplum Database をベースとするオンライン MPP (大規模な並列処理) データウェアハウスサービス。AnalyticDB for PostgreSQL は、マシン動作中におけるスケールアップ機能とパフォーマンスモニタリング機能を提供しており、MPP クラスターの運用と管理 (O&M) に伴う複雑な作業は不要です。このため、データベース管理者、開発者、データアナリストは、開発に専念し、生産性を向上することができます。

https://www.alibabacloud.com/ja/product/hybriddb-postgresql

# PythonLoader

https://github.com/hwchase17/langchain/pull/3311

Pythonファイルのエンコーディングを自動検出する機能

# おわりに
以上が2023/4/23のLangChainのアップデートでした．

記事に誤り等ありましたらご指摘いただけますと幸いです。
