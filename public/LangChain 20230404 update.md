---
title: LangChain 20230404 update
tags:
  - ChatGPT
  - langchain
  - LLaMA
  - GPT-4
  - GPT4All
private: false
updated_at: '2023-04-24T00:54:40+09:00'
id: 4a2fe7fad49f8702e2ec
organization_url_name: null
slide: false
ignorePublish: false
---
LangChain v0.0.131

# はじめに
LangChainの2023/4/4のupdate情報です．新しく以下の機能が追加されました．
* GPT4all
* LLaMA.cpp
* Qdrant
* Zilliz
* Outlook Message Loader
* Selenium Document Loader


https://twitter.com/LangChainAI/status/1643261943803957249?s=20

記事に誤り等ありましたらご指摘いただけますと幸いです。

備考：LangChainの実装まとめ
https://github.com/fuyu-quant/langchain

# 目次
- [updateの概要](#updateの概要)
- [GPT4all](#gpt4all)
- [LLaMA.cpp](#llamacpp)
- [Qdrant](#qdrant)
- [Zilliz](#zilliz)
- [Outlook Message Loader](#outlook-message-loader)
- [Selenium Document Loader](#selenium-document-loader)
- [おわりに](#おわりに)


# updateの概要

LangChainには以下にあるように大きく6つのモジュールで構成されています．

太字の箇所が今回アップデートされた箇所になります．

- **Models**
    - **LLMs**
    - **Text Embedding Models**
- Prompts
- **Indexes**
    - **Document Loaders**
    - **VectorStore**
- Memory
- Chains
- Agents

　今回のアップデートではModelsの中のLLMsという様々な大規模言語モデルを使うための標準的なインターフェースに**GPT4all**と**Llama-cpp**が追加されました．Llama-cppについては文章の埋め込み表現を得るためのText Embedding Modelsにも追加されました．

　またIndexesというLLMがドキュメントを様々な形でやりとりするために構造化するためのモジュールの中のVectorStoreという主に検索などを行う機能が集まっているものに**Qdrant**と**Zilliz**が追加されました．そして同じIndexesの様々な文章を読み込む機能が集まっているDocument Loadersに**Outlook Message Loader**と**Selenium Document Loader**が追加されました．

以下ではそれぞれについて説明します

# GPT4all
LLMs

https://python.langchain.com/en/latest/modules/models/llms/integrations/gpt4all.html

GPT4allはLLaMaをベースにし，GPT3.5-Turboで生成した文章を学習に使ったモデルです．量子化を行っているため手元のパソコンでも動かせるほど軽量です．

https://github.com/nomic-ai/gpt4all

以下は実行のためのコードです．

```python
!pip install llama-cpp-python

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# ここのモデルの重みは上のgithubのリンクからダウンロードしてくる必要があります
llm = LlamaCpp(model_path="./ggml-model-q4_0.bin")
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = ""

llm_chain.run(question)
```

手元で実行したところ，英語では正しい回答をある程度の速度で生成することができていましたが日本語での性能はかなり悪くまともに返してくれませんでした．他の検証記事にもあるように日本語は得意ではないみたいです.

- [https://gigazine.net/news/20230331-gpt4all-how-to-use/](https://gigazine.net/news/20230331-gpt4all-how-to-use/)

# LLaMA.cpp
LLMs

https://python.langchain.com/en/latest/modules/models/llms/integrations/llamacpp.html

Text embedding models

https://python.langchain.com/en/latest/modules/models/text_embedding/examples/llamacpp.html

LLaMA.cppはC/C++による実装で4ビット量子化によりGPT4all同様に手元のパソコンで動くようなモデルです．

推論に必要な部分の重みだけがユーザーのメモリにロードされた結果，メモリ使用量の減少によって推論の時間が非常に早くなったそうです．またメモリ使用量がなぜ少なくなるのかは正確にわかっていないそうです．

https://github.com/ggerganov/llama.cpp

以下が実行コードです．

```python
!pip install llama-cpp-python

# 通常のLLMとしての利用

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# モデルの重みをダウンロードしてくる必要があります
llm = LlamaCpp(model_path="./ggml-model-q4_0.bin")
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = ""

llm_chain.run(question)

# Text Embedding

from langchain.embeddings import LlamaCppEmbeddings

# モデルの重みをダウンロードしてくる必要があります
llama = LlamaCppEmbeddings(model_path="/path/to/model/ggml-model-q4_0.bin")
text = "This is a test document."
query_result = llama.embed_query(text)
doc_result = llama.embed_documents([text])
```

- [https://gigazine.net/news/20230403-llama-cpp-ram/](https://gigazine.net/news/20230403-llama-cpp-ram/)

# Qdrant
Vectorstores

https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/qdrant.html
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/langchain/blob/main/indexes/vectorstores/qdrant.ipynb)
Qdrantというベクトル検索とベクトルデータベースのサービスを使った機能です．検索元のテキストの埋め込み表現と検索クエリのテキストの埋め込み表現を入力することで検索元のテキストから類似度の高いテキストを検索することができます．

さらに，Qdrantでは検索元のテキストとその埋め込み表現の保存場所は，**メモリ上，ディスクのストレージ，オンプレサーバー，Qdrantクラウド**など様々なものを選ぶことができ非常に汎用性が高いです．

[https://qdrant.tech/](https://qdrant.tech/)

以下はメモリ上にテキストと埋め込み表現を保存して，検索クエリに類似しているものを取り出すための実装です．

検索の方法にも以下の3種類が用意されています

- 類似文章の検索
- スコア付きの類似文章の検索
- Maximum marginal relevance search(MMR)…似た文章だけでなく多様な文章を取得できる

```python
!pip install langchain
!pip install openai
!pip install qdrant-client

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader

# 検索元にしたいテキストを読み込ませる
loader = TextLoader('sample.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

qdrant = Qdrant.from_documents(
    docs, embeddings, 
    location=":memory:",  
    collection_name="my_documents",
)

# 類似文章の検索
query = ""
found_docs = qdrant.similarity_search(query)

# スコア付きの類似文章の検索
query = ""
found_docs = qdrant.similarity_search_with_score(query)
document, score = found_docs[0]
print(document.page_content)
print(f"\nScore: {score}")

# Maximum marginal relevance search
query = ""
found_docs = qdrant.max_marginal_relevance_search(query, k=2, fetch_k=10)
for i, doc in enumerate(found_docs):
    print(f"{i + 1}.", doc.page_content, "\n")
```

検索元の保存場所や検索方法が複数用意されていて非常に便利だと感じました．

# Zilliz
Vectorstores

https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/zilliz.html

Zilliz クラウドを使いベクトル検索を行うための機能です．現在はZillizaとの連携が前提の機能のみなのでZillizを活用されている人に役に立ちそうです．

[Vector database platform - Zilliz Cloud](https://zilliz.com/cloud)

実装は以下になります．

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader

# Zillizとの連携に必要
ZILLIZ_CLOUD_HOSTNAME = ""
ZILLIZ_CLOUD_PORT = ""

loader = TextLoader('sample.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": ZILLIZ_CLOUD_HOSTNAME, "port": ZILLIZ_CLOUD_PORT},
)

docs = vector_db.similarity_search(query)
docs[0]
```

# Outlook Message Loader
Document Loaders

https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/email.html

メールのファイルからテキストを取得するための機能です．今回のアップデートにより(.eml)に加えてMicrosoft Outlook(.msg)のファイルからもテキスト情報を取得できるようになりました．

実装は以下です．

```python
from langchain.document_loaders import OutlookMessageLoader

loader = OutlookMessageLoader('sample.msg')
data = loader.load()

data[0]
```

# Selenium Document Loader
Document Loaders

https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/url.html

URLのリストを受け取りHTMLから文章の情報だけを取得するための機能．今回新しく加わったSelenium Document LoaderによりレンダリングにJavaScriptを必要とするページから読み取ることができるようになりました．

```python
!pip install langchain
!pip install unstructured
!pip install selenium

from langchain.document_loaders import SeleniumURLLoader

urls = []

loader = SeleniumURLLoader(urls=urls)
data = loader.load()
data
```

# おわりに

以上が2023/4/4のLangChainのアップデートでした．

個人的にはQdrantが非常に便利で様々なことに応用できそうだと感じました．

記事に誤り等ありましたらご指摘いただけますと幸いです。
