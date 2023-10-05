---
title: ChatGPTとLangChainで便利な検索ツールを作る
tags:
  - OpenAI
  - ChatGPT
  - langchain
  - 記事投稿キャンペーン_ChatGPT
  - LlamaIndex
private: false
updated_at: '2023-06-19T16:26:23+09:00'
id: 098a3aeb3c5ae5b290cd
organization_url_name: brainpad
slide: false
ignorePublish: false
---
　この記事は[記事投稿キャンペーン_ChatGPT](https://qiita.com/official-events/b80ebe828fa453edf260)の記事です。

以下は、何でもできるAIをコンセプトに個人開発したものです。
よかったら見てみてください。
[CreateToolAGI：ChatGPTとLangChainで何でもできるAI](https://qiita.com/fuyu_quant/items/5e5b4f429e775ac39a74)


# はじめに
こんにちは、[fuyu-quant](https://twitter.com/fuyu_quant)です!
　今回はLangChainやllama-indexなどのOSSを使い**URL vector search**という，**URLを与えるだけでベクトルデータベースを作成**し，**質問を与えると類似している内容のURLとそのリンク先ごとに質問内容を踏まえた説明を出力**してくれるツールを作成しました．

　まとめサイトや個人のブログなどでChatGPTを使った検索や内容の解説をさせたりする際に参考になるかと思います．実装内のプロンプトを書き換えればそれぞれのサイトにあったものが構築できると思います．

記事に誤り等ありましたらご指摘いただけますと幸いです。

(※この記事の「ChatGPT」は「ChatGPT API」の意味になります)

以下が今回の作成したものになります。

https://github.com/fuyu-quant/url-vector-search

# 目次
- [1. 実行方法](#1-実行方法)
- [2. 使用したライブラリ](#2-使用したライブラリ)
- [3. 各機能の実装方法](#3-各機能の実装方法)
    - [URLからデータを取得](#urlからデータを取得)
    - [テキストデータの要約](#テキストデータの要約)
    - [検索データベースの作成](#検索データベースの作成)
    - [検索の実行](#検索の実行)
- [4. おわりに](#4-おわりに)
- [5. 参考文献](#5-参考文献)


# 1. URL vector searchについて

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/url-vector-search/blob/main/examples/urlvectorsearch.ipynb)

　今回作成したツールは、与えたURLに対してリンク先のテキストデータにもとづいて検索を行い類似している内容のURLと検索内容を考慮した説明文の出力を行う事ができます．
(※実行にはOpeaAIのAPIが必要です)

今回は以下のサイトのリンクを使っています。

https://www.brainpad.co.jp/doors/

また完全に告知になるのですが会社で以下のようなイベントを実施予定です。
データ活用の先端事例について学びたい方は是非見てみてください!!

https://www.brainpad.co.jp/doors/conference23/?utm_source=analytics&utm_medium=email&utm_campaign=an9

使い方は以下のようになります。

まずはじめに検索対象となるベクトルデータベースを作成します．

```python
# 好きなパスを指定してください
path = '/content/qdrant'
database = vectordatabase.create_vectordatabase(path = path)
```

次に検索対象とするリンクを作成したデータベースに与えます．

(検索はURLごとの検索になります．データベースの作成には少し時間がかかりますが一度追加したものは永遠に保存されるので新しいものを追加したい時だけ実行することになります)

```python
urls = [
    "https://www.brainpad.co.jp/doors/knowledge/01_dx_natural_language_processing_1/",   
    "https://www.brainpad.co.jp/doors/knowledge/01_quantitative_evaluation_year_2024_problem/",   
    "https://www.brainpad.co.jp/doors/news_trend/logistics_industry_year_2024_problem/",   
    "https://www.brainpad.co.jp/doors/news_trend/materials_informatics/", 
    .......
    "https://www.brainpad.co.jp/doors/feature/02_conference_2021_crosstalk_session_resona_2/"
]

vectordatabase.add_text(urls, database)
```

これで作成は完了しました。

以下のように質問をすると，

```python
query = "機械学習"
output1 = output.description_output_with_scoer(database, query, num_output=3)
```

質問内容に関連するリンクと質問内容を踏まえた内容が出力されます。
(※今回は10種類ほどしかテキストを与えてないので少しずれている内容も出力されました．)

```python
# 出力結果
output1

{'descritption0': 'この記事は、梅田氏による自然言語処理のビジネス活用についての指摘や、...',
 'score0': 0.8361707487956747,
 'url0': 'https://www.brainpad.co.jp/doors/knowledge/01_dx_natural_language_processing_1/',
 'descritption1': 'この記事は、ブレインパッドの3人がデジタル技術を用いて物量平準化...',
 'score1': 0.833812125722534,
 'url1': 'https://www.brainpad.co.jp/doors/knowledge/01_quantitative_evaluation_year_2024_problem/',
 'descritption2': 'この記事は、OpenAI社が開発したChatGPTという対話型AIについて紹介...',
 'score2': 0.8233397829398639,
 'url2': 'https://www.brainpad.co.jp/doors/news_trend/about_chatgpt/'}
```

他にも何種類かの出力方法を用意しています．

現在の実装だと検索結果を３つ出力するのに30秒ほどかかってしまいますが，時間がかかっている箇所はLLMの実行によるもので並列化可能なので少なくとも10秒ぐらいまでにはできそうです．






# 2. 使用したライブラリ

- LangChain
    - LLMとの連携をする上でとても使いやすいので使用しています。今後OpenAIのLLM以外のLLMを使う際も簡単に切り替えられるようになっていくと思っているのでその点でも採用しています。
- Qdrant
    - ベクトル検索とベクトルデータベースが使いやすいのとLangChainからも使えるので採用しました。
- llama-index
    - LangChainのURL toolがうまく実行できなかったためURLからの情報の取得にはllama-indexを使いました．
- OpenAI
    - OpenAIのLLMを利用するため使っています。

各ツールのバージョン

```python
!pip install langchain==0.0.153
!pip install llama-index==0.5.27
!pip install qdrant-client==1.1.6
!pip install openai==0.27.5
```



# 3. 各機能の実装方法

今回作成したもののポイントとなる実装方法を紹介します．

## URLからデータを取得

llama_indexの機能を使いURLを渡し，テキスト情報の抽出を行います．

今回は以下のページの情報を取得してきます．

https://www.brainpad.co.jp/doors/knowledge/01_dx_natural_language_processing_1/

```python
from llama_index.readers import BeautifulSoupWebReader

urls = [
    "https://www.brainpad.co.jp/doors/knowledge/01_dx_natural_language_processing_1/",    
    "https://www.brainpad.co.jp/doors/knowledge/01_quantitative_evaluation_year_2024_problem/"
]

documents = BeautifulSoupWebReader().load_data(urls=urls)
```

取得結果は以下のようなものになります．

```python
# 1つ目のリンクに関する情報
documents[0]

# 出力結果
Document(text='\n\n\n\n\n\n\n\n\n\n \n\n\n\n\n【前編】自然言語処理技術を用いた「テキストデータ」
のビジネス活用の現在地 | DX・データ活用情報発信メディア-DOORS DX\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n
\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n \n\n\n\n\n\nベストなDXへの入り口が見つかるメディア
......
```

## テキストデータの要約

取得したデータを要約します．これは以下の２点の理由から実施しました．

- 検索クエリに対して内容を反映した出力を生成する際にテキストデータが長くなるとLLMに入力する事ができなくなる
- 検索後にヒットしたものを要約するようにすると最終的な出力を生成するのに時間がかかってしまう

ここの項目は，「テキストデータが長くない場合」や「時間はかけて良いが正確な出力が欲しい場合」は省力してよいと考えています．

今回は以下のページについて要約を行います．

https://www.brainpad.co.jp/doors/knowledge/01_dx_planning_poc_phase/


テキストデータを分割し，LLMに入力するためのデータ形式に変換する

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

text =  "機械学習プロジェクトを推進するにあたって大切なこと～DX推進時...."

# 1000文字ごとに'\n\n'を挿入する
chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
result = '\n\n'.join(chunks)

# '\n\n'で文字列を分割し一つのチャンクとする
text_splitter = CharacterTextSplitter(
    separator = "\n\n",  # 文章を分割する文字列
    chunk_size = 1000,  # チャンクの文字数
    chunk_overlap = 0,  # チャンク間で重複させる文字数
)

# 分割した文字列をLLMに入力するデータ形式に変換する
split_texts = text_splitter.split_text(result)
split_docs = [Document(page_content=t) for t in split_texts]
```

今回は"map_reduce"という方法を使い，二段階に分けて要約を行うのでそれぞれの段階でのプロンプトを作成します．

map_reduceについてはこちらのリンクを参考にしてみてください．

https://note.com/mega_gorilla/n/n6f46fc1985ca

```python
from langchain import PromptTemplate

template1 = """
次の文章を日本語で500文字程度に要約してください．
文章：{text}
"""

template2 = """
次の文章を日本語で1200文字程度に要約してください．
文章：{text}
"""

prompt1 = PromptTemplate(
    input_variables = ['text'],
     template = template1,
)

prompt2 = PromptTemplate(
    input_variables = ['text'],
     template = template2,
)
```

要約の実行

```python
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(temperature=0, max_tokens = 1200)

chain = load_summarize_chain(
    llm = llm, 
    chain_type="map_reduce",
    # それぞれの要約を行うときのテンプレ
    map_prompt = prompt1,
    # 要約文の要約文を作るときのテンプレ
    combine_prompt = prompt2
    )

summary = chain.run(split_docs)

summary

# 出力結果
ブレインパッドは、機械学習プロジェクトを推進するため、企画・PoCフェーズを重要視しています。企画フェーズでは、
プロジェクト全体の目的とPoCの目的をお客様と弊社ですり合わせ、PoCの目的で多いのは機械学習モデルの精度検証
であることを確認します。また、プロジェクトを始める前に知っておくべきことや、プロジェクトの進め方のイメージを実体験
に基づきお伝えしています。PoCフェーズでは、企画フェーズで設定した目的に沿って実際に検証を実施し、
機械学習モデルの性能がビジネスに使えるレベルまで達しているかを確認します。アノテーションの重要性も改めて
感じることができます。プロジェクトを取り組む際には、精度を上げることが必ずしも正しいとは限らないことを
理解しておくことが重要です。
```

今回は以上のような方法で取得したテキストを要約しましたが，設定や要約の方法を変えることでよりもとの文章の情報を失わずに要約できる方法があると思うのでさらに改善できる箇所かと思います．

## 検索データベースの作成

検索対象の文章を保存するためのベクトルデータベースを作成します．今回は以下の理由からQdrantというベクトルデータベースのサービスを使います．

- 外部APIを使わなくてもベクトルデータベースを作成できる
- さまざまな保存場所を選ぶ事ができる
    - https://qiita.com/fuyu_quant/items/4a2fe7fad49f8702e2ec#qdrant
- LangChainで実装できる
    - https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/qdrant

Qdrantのベクトルデータベースの作成は

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document

embeddings = OpenAIEmbeddings()

# サンプルのデータを
sample = [Document(page_content="sample", metadata={"url": "aaaa" })]
# ベクターインデックスを作成する
qdrant = Qdrant.from_documents(sample, embeddings, location=":memory:", collection_name= "my_documents")
```

次に以下のページの情報を登録してみます

https://www.brainpad.co.jp/doors/knowledge/01_data_governance_9_1/

二つ目以降のデータの登録は以下のように行います．

```python
text = "データドリブン文化醸成とは（後編） | DOORS DX ベストなDXへの入り口が見つかる...."

# textをLangChainで扱えるようにデータ形式を変換する
# metadataの登録はしてもしなくても良い
docs = [Document(page_content=text2, metadata={"url": "https://www.brainpad.co.jp/doors/knowledge/01_data_governance_9_1/" })]

qdrant.add_documents(docs, collection_name="my_documents")
```

## 検索の実行

上記の方法で作成したベクトルデータベースに対して検索を行います．検索方法にはスコア付きの出力を得るものやMMRという類似度の高さ以外に取得するデータの多様性を高める方法などがあります．

実行結果は以下のようになります

```python
query = "データドリブン文化醸成とは"

# 類似度スコア付きの出力
# 類似度スコアが高いものから順番に取得してきます
found_docs = qdrant.similarity_search_with_score(query)
document, score = found_docs[0]
print(document.page_content)
print(f"\nScore: {score}")

# 出力結果
データドリブン文化醸成とは（後編） | DOORS DX ベストなDXへの入り口が見つかる....
Score: 0.9071070813602901

# スコアなしの出力
# 類似度が高いものから順番に取得してきます
found_docs = qdrant.similarity_search(query)
print(found_docs[0].page_content)

# 出力結果
データドリブン文化醸成とは（後編） | DOORS DX ベストなDXへの入り口が見つかる....

# MMR(Maximal Marginal Relevance)
# 類似度が高いものを取得したいがなるべく多様な結果も取得したいときに使う
found_docs = qdrant.max_marginal_relevance_search(query, k=2, fetch_k=10)
print(found_docs[0].page_content)

# 出力結果
データドリブン文化醸成とは（後編） | DOORS DX ベストなDXへの入り口が見つかる....
```

- MMRについては以下のサイトなどでも紹介されています

https://yolo-kiyoshi.com/2020/05/08/post-1781/#outline__2



# 4. おわりに

　最後までお読みいただきありがとうございます。
今回は**任意のURLの内容についての検索や質問内容を踏まえたテキストベースの出力を得るためのツール**を作成しました．現状では元の文章を要約しているのであまり元の内容が反映できていないものがあったりするので，そのあたりは改善の余地があると考えています．改善案などがあればぜひ教えていただきたいです!

記事に誤り等ありましたらご指摘いただけますと幸いです。




# 5. 参考文献
- https://note.com/mega_gorilla/n/n6f46fc1985ca
- https://yolo-kiyoshi.com/2020/05/08/post-1781/#outline__2
