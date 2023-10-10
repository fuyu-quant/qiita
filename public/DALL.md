---
title: DALL・E(API)
tags:
  - OpenAI
  - diffusionmodel
  - DALL-E
private: false
updated_at: '2023-10-11T02:53:12+09:00'
id: 1c382302c4c8311b98e9
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

今回はDALL・E(API)の紹介になります．現在，ChatGPT上で一部の人?が使えるDALL・E3が流行っていますが，今回は現在利用できるDALL・EのOpenAI APIの使い方についてまとめました．
記事に誤り等ありましたらご指摘いただけますと幸いです。

https://openai.com/dall-e-2

# 目次
- [1. DALL・E(API)](#1-dall・e(api))
- [2. 使い方](#2-使い方)
    - [画像生成](#画像生成)
    - [類似画像の生成](#類似画像の生成)
    - [画像編集](#画像編集)
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)

# 1. DALL・E(API)

ライセンス:[Modified MIT License](https://github.com/openai/DALL-E/blob/master/LICENSE)
公式サイト:https://platform.openai.com/docs/guides/images
デモサイト:https://labs.openai.com/
論文:https://cdn.openai.com/papers/dall-e-2.pdf

DALL・E のAPI版の紹介です．APIで使われているモデルは明記はされていないですがDALL・E2と記載されている記事が多いので，内部ではDALL・E2が動いていると思われます．DALL・E3のAPIも早く利用したいですね．

https://openai.com/dall-e-2

# 2. 使い方

すぐに試したい方は各セクションのData Science WikiのページまたはColabのリンクから実行してみてください．

以下が今回使うライブラリです．(OpenAI のAPI keyも必要です)

```python
import openai
import requests
from PIL import Image
from io import BytesIO
import IPython.display as display
```

### 画像生成
まず初めは，プロンプトを入力して画像を生成してみたいと思います．

<a href="https://www.data-science-wiki.net/article?path=/multimodal/text_to_image/dall-e.html"><img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="300"/>
</a>

<a href="https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/multimodal/text_to_image/dall-e.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

画像生成

```python
response = openai.Image.create(
  prompt="a white siamese cat",
  n=1, # 1〜10枚まで設定できる
  # 三つのサイズを選択できる
  #size = "256x256",
  #size = "512x512",
  size="1024x1024"
)
```

生成した画像の取得

```python
image_url = response['data'][0]['url']

response = requests.get(image_url)
response.raise_for_status()
```

画像の表示

```python
image_data = BytesIO(response.content)
image = Image.open(image_data)
display.display(image)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/c1bc9fa6-2843-b36c-6b49-7198c7e26013.png)


画像の保存

```python
with open('sample_image.png', 'wb') as file:
    file.write(response.content)
```

### 類似画像の生成

<a href="https://www.data-science-wiki.net/article?path=/multimodal/Image_editing/dall-e_variations.html"><img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="300"/>
</a>

<a href="https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/multimodal/image_editing/dall-e_variations.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

先ほどの画像を元に新しい画像を生成します．

```python
response = openai.Image.create_variation(
  image=open("sample_image.png", "rb"),
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
```

画像の取得

```python
image_url = response['data'][0]['url']
response = requests.get(image_url)
response.raise_for_status()
```

新しく作成した類似画像の表示

```python
image_data = BytesIO(response.content)
image = Image.open(image_data)
display.display(image)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/39e86185-d7ca-04fb-cc2a-ad2431df5b8e.png)


### 画像編集
ここではせっかく真っ白な猫ですが，顔だけを黒い猫にしたいと思います．

<a href="https://www.data-science-wiki.net/article?path=/multimodal/Image_editing/dall-e.html"><img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="300"/>
</a>

<a href="https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/multimodal/image_editing/dall-e.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

先ほどの画像の猫の顔の部分をマスクします．

```python
image = Image.open('sample_image.png')
image = image.convert("RGBA")

# 画像の一部を透明にするための描画オブジェクトを作成します
draw = ImageDraw.Draw(image)

# 透明にする矩形領域を指定します (left, top, right, bottom)
rect = (300, 10, 1000, 600)

# 透明な四角形を描きます
draw.rectangle(rect, fill=(255, 255, 255, 0))  # 最後の0は透明度を示す

image.save('mask.png', "PNG")
display.display(image.convert("RGB"))
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/a06941a8-4a4b-98ba-0a92-654c56ec71d0.png)

画像をプロンプトによる指示を行い編集します．

```python
response = openai.Image.create_edit(
  image=open("sample_image.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="Cat with black head and blue eyes.",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
```

画像の表示

```python
image_data = BytesIO(response.content)
image = Image.open(image_data)
display.display(image)
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/0493bc56-2e9e-5839-937c-18dbb6d4301f.png)

指定した箇所だけプロンプトで指示した通りに編集することができました．編集後のクオリティもかなり高いですね．

# 4. おわりに

今回はDALL・E(API)の紹介でした．画像生成のクオリティとしては非常に高いですが無料の画像生成も多い中なかなか割高な気もします．DALL・E3のAPI提供されより安価になって欲しいものです．

記事に誤り等ありましたらご指摘いただけますと幸いです。

# 5. 参考文献
- https://signal.diamond.jp/articles/-/1492
