---
title: 'BLIP,BLIP-2'
tags:
  - MultiModal
  - huggingface
  - LLM
  - BLIP-2
  - BLIP
private: false
updated_at: '2023-09-22T17:38:34+09:00'
id: 50e0b2a9a0b5947527c6
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

こんにちは、[fuyu-quant](https://twitter.com/fuyu_quant)です．

今回はBLIP,BLIP-2の紹介になります．BLIPは2022年1月にSalesforceにより論文として発表された新しいVision Language Pre-training(画像と自然言語を使った事前学習)のフレームワークです．Image Captioning(画像からテキストを生成)やVisual question answering(画像への質問)を行うことができます．またBLIP-2はその後続の研究で学習済みのVision modelやLLMを使えるように改良しSOTAを達成しています．
記事に誤り等ありましたらご指摘いただけますと幸いです。

BLIP

https://github.com/salesforce/BLIP

BLIP-2

https://github.com/salesforce/LAVIS/tree/main/projects/blip2

# 目次
- [1. BLIP](#1-blip)
- [2. BLIP-2](#2-blip-2)
- [3. 使い方(BLIP)](#3-使い方(blip))
- [4. 使い方(BLIP-2)](#4-使い方(blip-2))
- [5. 日本語モデル(BLIP)](#5-日本語モデル(blip))
- [6. おわりに](#6-おわりに)
- [7. 参考文献](#7-参考文献)



# 1. BLIP

ライセンス：モデルを学習した際のデータセットによるのでそれぞれ確認してみてください．
リポジトリ:https://github.com/salesforce/BLIP
論文:https://arxiv.org/abs/2201.12086v1

BLIPはBootstrapping Language-Image Pre-trainingの略で画像と自然言語の両方を使った学習方法の新しいフレームワークです．ノイズの多いwebデータを効率的に利用するためノイズを除去するブートストラップを利用しています．詳細については論文を参照してください．

# 2. BLIP-2

ライセンス:モデルを学習した際のデータセットによるのでそれぞれ確認してみてください．
リポジトリ:https://github.com/salesforce/LAVIS/tree/main/projects/blip2
論文:https://arxiv.org/abs/2301.12597

BLIP-2では事前に学習された画像のモデルとLLMのパラメータを固定し，Q-Formerという変換器を追加したアーキテクチャでありQ-Formerが学習可能なパラメータを持っている．学習済みのモデルを利用することでBLIPに比べて必要な学習パラメータが大幅に少ないにも関わらずSOTAを達成しています．

BLIPおよびBLIP-2はsalesforce-lavisというライブラリからも利用することができるが，今回はhugging faceを使い実装する．

salesforce-lavisについて以下を確認してみてください．

https://github.com/salesforce/LAVIS

# 3. 使い方(BLIP)
すぐに試したい方は以下のリンクから実行してみてください

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/multimodal/text_image/BLIP(japanese).ipynb)

今回は以下の画像についてImage captioningとVisual question answeringを行っていこうと思います．
![tokyo_tower.jpeg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/4b6fe37f-2187-f078-a0a4-37e0037d08b4.jpeg)

以下のコマンドで必要なライブラリをインストールする．

```python
!pip install sentencepiece transformers
```

```python
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from PIL import Image
import requests
```

以下のコマンドでモデルのダウンロードを行う．

```python
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

## ****Image Captioning****

画像に対する説明文を取得したいと思います．実装は以下のコードです．

```python
inputs = processor(image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# 出力
arafed view of a city at dusk with a tall tower
```

arafedがよく意味の分からない単語ですが，後半部分は大体正しい感じかします．

## ****Visual question answering****

以下は画像に対して質問を行うためのコードです．

```python
prompt = "a photography of"

inputs = processor(image, text = prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# 出力
a photography of a cityscape with a tall tower in the middle
```

大体は画像の特徴を捉えた回答をしています．

# 4. 使い方(BLIP-2)

すぐに試したい方は以下のリンクから実行してみてください

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/multimodal/text_image/BLIP2.ipynb)

BLIP同様，以下のコマンドで必要なライブラリをインストールする．
```python
!pip install accelerate sentencepiece transformers
```

```python
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

from PIL import Image
import requests
```

モデルのダウンロード

```python
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

先ほどと同じ画像についてImage captioningとVisual question answeringを行っていきたいと思います．

## ****Image captioning****

画像に対する説明文を取得したいと思います．実装は以下のコードです．

```python
inputs = processor(image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# 出力
tokyo skyline at dusk, japan
```

東京タワーの画像から場所が日本であることを認識できているみたいです．

## ****Visual question answering****

以下は画像に対して質問を行うためのコードです．

```python
question = "What is the main thing in the picture?"
prompt = f"Question: {question} Answer:"

inputs = processor(image, text = prompt, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# 出力
Tokyo skyline
```

Tokyo towerと答えさせたかったですがここではskylineという回答になりました．

# 5. 日本語モデル(BLIP)

日本語でBLIPを使いたい場合はStability AIが作成した以下のモデルが利用可能です．(現状では商用利用は不可のようです(2023/9/18))

https://ja.stability.ai/blog/japanese-instructblip-alpha

コードについては以下をご利用ください．

https://github.com/fuyu-quant/data-science-wiki/blob/main/multimodal/text_image/BLIP(japanese).ipynb

# 6. おわりに

今回はBLIP,BLIP2の紹介でした．Image captioning(画像からの説明文生成)およびVisual question answering(画像への質問に対する回答)ともにBLIP,BLIP-2で回答できていましたがBLIP-2の方がより詳細に回答できている印象でした．BLIP-2では画像のモデルやLLM別々で学習を行った強いモデルを使えるので高い精度が出せていると考えられます．
記事に誤り等ありましたらご指摘いただけますと幸いです。

# 7. 参考文献

- https://qiita.com/m__k/items/a465df3283fe3ffc8e31
- https://www.12-technology.com/2022/02/clip-blip.html
- https://blog.shikoan.com/blip-2/
- https://huggingface.co/blog/blip-2
