---
title: LangChain 20230405 update
tags:
  - ChatGPT
  - langchain
  - GPT-4
private: false
updated_at: '2023-04-24T00:54:58+09:00'
id: f2f9253f3d2a2cc3af38
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

LangChainの2023/4/5のupdate情報です．新しく以下の機能が追加されました．

- TF-IDF Retriever
- ElasticSearch BM25 Retriever
- Pinecone Hybrid Search
- Metal

https://twitter.com/LangChainAI/status/1643628476505681920?s=20

記事に誤り等ありましたらご指摘いただけますと幸いです．

備考：LangChainの実装まとめ
https://github.com/fuyu-quant/langchain

# 目次
- [updateの概要](#updateの概要)
- [TF-IDF Retriever](#tf-idf-retriever)
- [ElasticSearch BM25 Retriever](#elasticsearch-bm25-retriever)
- [Pinecone Hybrid Search](#pinecone-hybrid-search)
- [Metal](#metal)
- [おわりに](#おわりに)

# updateの概要

LangChainには以下にあるように大きく6つのモジュールで構成されています．

太字の箇所が今回アップデートされた箇所になります．

- Models
- Prompts
- **Indexes**
    - **Retrievers**
- Memory
- Chains
- Agents

今回はIndexesのRetrieversに4つのアップデートがありました．Retrieversは文字列を受け取り，文章のリストを返す機能のものが集められています．具体的には類似する文章の検索結果を返すようなものがあります．

# TF-IDF Retriever

Retrievers

https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/tf_idf_retriever.html

scikit-learnのTF-IDFを使いテキストの中から類似度が高いものを検索することができます．

※動作は確認できていません

```python
!pip install scikit-learn
!pip install langchain==0.0.134

from langchain.retrievers import TFIDFRetriever

retriever = TFIDFRetriever.from_texts(["foo", "bar", "world", "hello", "foo bar"])

result = retriever.get_relevant_documents("foo")
result
#[Document(page_content='foo', metadata={}),
# Document(page_content='foo bar', metadata={}),
# Document(page_content='hello', metadata={}),
# Document(page_content='world', metadata={})]
```

# ElasticSearch BM25 Retriever

Retrievers

https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/elastic_search_bm25.html

ElasticSearchとBM25を利用するretrieversです．

ElasticSearchは分散検索/分析エンジンで大容量データを処理することを想定した全文検索エンジンでBM25は検索アルゴリズムに使われる自然言語処理の一つでtf-idfを発展させたようなものになっています．

ElasticSearch BM25 Retrieverを使うことでテキストから類似度の高いものを検索することができます

```python
!pip install elasticsearch
!pip install langchain==0.0.134

from langchain.retrievers import ElasticSearchBM25Retriever

elasticsearch_url="http://localhost:9200"
retriever = ElasticSearchBM25Retriever.create(elasticsearch_url, "langchain-index-4")

# 検索対象のテキストを追加
retriever.add_texts(["foo", "bar", "world", "hello", "foo bar"])

# "foo"を含むテキストを検索
result = retriever.get_relevant_documents("foo")

result
#[Document(page_content='foo', metadata={}),
# Document(page_content='foo bar', metadata={})]
```

参考リンク

- [BM25を数式から説明する](https://qiita.com/KokiSakano/items/2a0f4c45caaa09cf1ab9)

# Pinecone Hybrid Search

Retrievers

https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/pinecone_hybrid_search.html

Pineconeは高次元ベクトルのデータを格納できる機械学習用のデータベースサービスでPinecone Hybrid SearchではEmbeddingsを行うモデルとtokenizerを与えることで検索対象のテキストをベクトル化しPineconeのベクトルデータベースに格納し，Hybrid SearchというPineconeのサービスを使い類似しているテキストの検索を行います．

```python
!pip install pinecone-client
!pip install langchain==0.0.134

from langchain.retrievers import PineconeHybridSearchRetriever
import pinecone 
from langchain.embeddings import OpenAIEmbeddings
from transformers import BertTokenizerFast

pinecone.init(
   api_key="...",  # API key here
   environment="..."  # find next to api key in console
)

index_name = "..."

pinecone.create_index(
   name = index_name,
   dimension = 1536,  # dimensionality of dense model
   metric = "dotproduct",
   pod_type = "s1"
)
index = pinecone.Index(index_name)

embeddings = OpenAIEmbeddings()
tokenizer = BertTokenizerFast.from_pretrained(
   'bert-base-uncased'
)

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, index=index, tokenizer=tokenizer)

retriever.add_texts(["foo", "bar", "world", "hello"])

result = retriever.get_relevant_documents("foo")
result[0]
#Document(page_content='foo', metadata={})
```

参考リンク

- [ベクトル特化型データベースサービス「Pinecone」でセマンティック・キーワード検索をやってみた](https://dev.classmethod.jp/articles/dive-deep-into-modern-data-saas-about-pinecone/)

# Metal

Retrievers

https://python.langchain.com/en/latest/modules/indexes/retrievers/examples/metal.html

MetalはEmbeddingのマネージドサービスです．Metalに検索対象のテキストを与えることで，すぐに類似テキストの検索をすることができるようになります．MetalのAPI keyの取得が必要です．

```python
!pip install metal_sdk
!pip install langchain==0.0.134

from langchain.retrievers import MetalRetriever
from metal_sdk.metal import Metal

API_KEY = ""
CLIENT_ID = ""
APP_ID = ""

metal = Metal(API_KEY, CLIENT_ID, APP_ID);

# 検索元のテキストを与える
metal.index( {"text": "foo1"})
metal.index( {"text": "foo"})

# 検索
retriever = MetalRetriever(metal, params={"limit": 2})
retriever.get_relevant_documents("foo1")

#[Document(page_content='foo1', metadata={'dist': '1.19209289551e-07', 'id': '642739a17559b026b4430e40', 'createdAt': '2023-03-31T19:50:57.853Z'}),
# Document(page_content='foo1', metadata={'dist': '4.05311584473e-06', 'id': '642738f67559b026b4430e3c', 'createdAt': '2023-03-31T19:48:06.769Z'})]
```

参考リンク

- [Metal](https://docs.getmetal.io/misc-create-app)

# おわりに

以上が2023/4/5のLangChainのアップデートでした．

記事に誤り等ありましたらご指摘いただけますと幸いです。
