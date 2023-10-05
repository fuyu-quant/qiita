---
title: LangChain 20230421 update
tags:
  - ChatGPT
  - langchain
  - GPT-4
private: false
updated_at: '2023-04-24T00:55:37+09:00'
id: 0338de4577a0aa857a2f
organization_url_name: null
slide: false
ignorePublish: false
---
ver 0.0.145

# はじめに

LangChainの2023/4/21のupdate情報です．新しく以下の機能が追加されました．

- Supabase VectorStore
- Discord Document Loader
- File Based Chat Message History
- Arxiv Tool
- Duck Duck Go Search Tool
- Google Places Tool

https://twitter.com/LangChainAI/status/1649125648525590528

記事に誤り等ありましたらご指摘いただけますと幸いです．

備考：LangChainの実装まとめ
https://github.com/fuyu-quant/langchain

# 目次
- [updateの概要](#updateの概要)
- [SupabaseVectorStore](#supabasevectorstore)
- [Discord Document Loader](#discord-document-loader)
- [File Based Chat Message History](#file-based-chat-message-history)
- [Arxiv Tool](#arxiv-tool)
- [Duck Duck Go Search Tool](#duck-duck-go-search-tool)
- [Google Places Tool](#google-places-tool)
- [おわりに](#おわりに)

# updateの概要

LangChainには以下にあるように大きく6つのモジュールで構成されています．

太字の箇所が今回アップデートされた箇所になります．

- Models
- Prompts
- Indexes
    - **Vectorstores**
    - **Document Loaders**
- Memory
- Chains
- Agents
    - **Tools**

IndexesのVectorstoresに「****SupabaseVectorStore****」というSupabaseとpgvectorを使ったベクトル検索のための機能とDocument LoadersにDiscordのデータを取得するための「Discord Document Loader」が追加されました．

AgentsのToolsに新たに３つのtool，「Arxiv Tool」「Duck Duck Go Search Tool」「Google Places Tool」が追加されました．

# SupabaseVectorStore

https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/supabase.html

SupabaseとpgvectorをVectorStoreとして使う機能

* Supabaseについて

https://qiita.com/kabochapo/items/6a2a391832825d17af7d

* pgvectorについて
Postgresのためのオープンソースベクトル類似性検索機能

https://github.com/pgvector/pgvector

# Discord Document Loader

https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/discord_loader.html

Discordのデータを取得することができる機能

以下の手順でデータを取得する

- ユーザー設定に移動し，「プライバシー・安全」を選択

![スクリーンショット 2023-04-24 0.25.52.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/61ff6c6a-ad7d-6ca9-e0d9-86d46a2d7e90.png)

- 下の方にスクロースすると以下の「データをリクエストする」からデータを取得できます

(データの取得には最大で30日かかる場合もあるそうです)

![スクリーンショット 2023-04-24 0.26.03.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/1e99e50b-3f34-7e48-b517-d841ca01bd39.png)

- Discordよりメールが配信され，そこからデータをダウンロードできます

```python
import pandas as pd
import os

path = input("Please enter the path to the contents of the Discord \"messages\" folder: ")
li = []
for f in os.listdir(path):
    expected_csv_path = os.path.join(path, f, 'messages.csv')
    csv_exists = os.path.isfile(expected_csv_path)
    if csv_exists:
        df = pd.read_csv(expected_csv_path, index_col=None, header=0)
        li.append(df)

df = pd.concat(li, axis=0, ignore_index=True, sort=False)

from langchain.document_loaders.discord import DiscordChatLoader

loader = DiscordChatLoader(df, user_id_col="ID")
print(loader.load())
```

# File Based Chat Message History

https://github.com/hwchase17/langchain/pull/3198

確認中(チャットの内容を永続化するためのオプション機能が追加されたそうです)

# Arxiv Tool

https://python.langchain.com/en/latest/modules/agents/tools/examples/arxiv.html

arxivのデータを取得するためのTool，現状だと論文全てではなく一部の情報しか取得できないみたいです．

```python
!pip install arxiv
!pip install langchain==0.0.145

from langchain.utilities import ArxivAPIWrapper

arxiv = ArxivAPIWrapper()

# MAMLの論文のデータを取得
docs = arxiv.run("1703.03400")
docs

# 出力
# 'Published: 2017-07-18\nTitle: Model-Agnostic Meta-Learning for Fast Adaptation 
# of Deep Networks\nAuthors: Chelsea Finn, Pieter Abbeel, Sergey Levine\nSummary: 
# We propose an algorithm for meta-learning that is model-agnostic, in the\nsense 
# that it is compatible with any model trained with gradient descent and\napplicable 
# to a variety of different learning problems, including\nclassification, regression, 
# and reinforcement learning. The goal of\nmeta-learning is to train a model on a 
# variety of learning tasks, such that it\ncan solve new learning tasks using only 
# a small number of training samples. In\nour approach, the parameters of the model 
# are explicitly trained such that a\nsmall number of gradient steps with a small 
# amount of training data from a new\ntask will produce good generalization performance 
# on that task. In effect, our\nmethod trains the model to be easy to fine-tune. 
# We demonstrate that this\napproach leads to state-of-the-art performance on two 
# few-shot image\nclassification benchmarks, produces good results on few-shot 
# regression, and\naccelerates fine-tuning for policy gradient reinforcement 
# learning with neural\nnetwork policies.'

# 著者での検索
docs = arxiv.run("Caprice Stanley")
docs

# 'Published: 2015-04-10\nTitle: Learning Compact Convolutional Neural Networks
# ...
# -- and can handle novel\nobjects not seen during training.'
```

# Duck Duck Go Search Tool

https://python.langchain.com/en/latest/modules/agents/tools/examples/ddg.html

[DuckDuckGO](https://duckduckgo.com/)という個人情報をトラッキングしない検索エンジンを使った検索を行うためのTool

```python
!pip install duckduckgo-search
!pip install langchain==0.0.145

from langchain.tools import DuckDuckGoSearchTool

search = DuckDuckGoSearchTool()

search.run("株式会社ブレインパッドとは?")

# 出力
#'株式会社ブレインパッドは、2023年3月28日開催の取締役会において、代表取締役の異動について決議いたしました。
#. 創業者である佐藤 清之輔、高橋 隆史が代表取締役を退任し、2023年7月1日付にて、関口 朋宏（現・取締役 
#執行役員CGO）が代表取締役に就任 ... ブレインパッド、Co-Creation型の新オフィスへ本社移転を完了、
# 5月2日より営業開始 株式会社ブレインパッドは、2022年3月24日（木）に発表のとおり、本社を六本木ティーキューブ
# （東京都港区六本木）へ移転し、2022年5月2日（月）より営業を開始したことをお知らせいたします。 ブレインパッドの
# 平均年収は721万円. ブレインパッドの 平均年収は721万円 です。. 去年の全国平均年収430万円より67.7%高い
# です。. 過去のデータを見ると581万円 (最低)から738万円 (最高)の範囲で推移しています。. この平均収入は
# 賞与を含んだ金額です (一部 ... ブレインパッドの評判③：激務. 超激務です。. 2～3年で退職する人が多く、
# 人材採用は「商品の仕入れ」の様な感覚です。. これは勿体ないですね。. 優秀な分析家は短期間で育つものでは
# ないので、長く勤めてもらった方が良いはずです。. 管理人. 待遇を ... ブレインパッド、代表取締役の異動に関す
# るお知らせ. 2023.03.08. 経営と現場が一体となれる、具体的なDX戦略の実現に向けて。. 国内最大級の視聴規模
# となるDXプレミアムイベント「DOORS BrainPad DX Conference 2023」6月5日より開催. 2023.02.28. 
#[株式会社TimeTechnologies ...'
```

# Google Places Tool

https://python.langchain.com/en/latest/modules/agents/tools/examples/google_places.html

Google MapのAPIの利用

```python
!pip install googlemaps
!pip install langchain==0.0.145

from langchain.tools import GooglePlacesTool

places = GooglePlacesTool()
places.run("Roppongi Hills Tokyo")

# 出力
# '1. Roppongi Hills\nAddress: 6-chōme-10-1 Roppongi, 
# Minato City, Tokyo 106-6108, Japan\nPhone: 03-6406-6000\nWebsite: 
# https://www.roppongihills.com/\n\n\n2. Tokyo City View\nAddress: 
# Japan, 〒106-0032 Tokyo, Minato City, Roppongi, 6-chōme−10−１, 
# Roppongi Hills Mori Tower, 52階\nPhone: 03-6406-6652\nWebsite: 
# https://tcv.roppongihills.com/\n\n'
```

# おわりに

以上が2023/4/21のLangChainのアップデートでした．
今回はArxiv APIが気になりました．使いこなせばデータ収集をするのに便利そうです．

記事に誤り等ありましたらご指摘いただけますと幸いです。
