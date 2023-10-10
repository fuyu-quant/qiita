---
title: LLaVA
tags:
  - CV
  - MultiModal
  - LLM
private: false
updated_at: '2023-10-11T05:31:07+09:00'
id: 2692198b65d9763b45a2
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

今回はLLaVA(Large Language and Vision Assistant)の紹介になります．LLaVAは画像のエンコーダーとLLMのLlama2を合わた新しいend-to-endの学習済みモデルで，GPT4-Vのオープンソースのようなモデルです．ScienceQAというデータセットでSOTAも達成しています．日本語にも対応しているみたいなので日本語で検証を行っています．
記事に誤り等ありましたらご指摘いただけますと幸いです。

https://github.com/haotian-liu/LLaVA

# 目次
- [1. LLaVA](#1-llava)
- [2. 使い方](#2-使い方)
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)

# 1. LLaVA

ライセンス:Apache-2.0
リポジトリ:https://github.com/haotian-liu/LLaVA
公式サイト:https://llava-vl.github.io/
論文:　
https://arxiv.org/abs/2310.03744
https://arxiv.org/abs/2304.08485

LLaVAはLarge Language and Vision Assistantの略で，画像を詳細なテキスト情報に変換することができます．ScienceQAというマルチモーダルのデータセットにおいてはstate-of-the-artを達成しています．また新しい点として，画像とテキストのインストラクションチューニングデータを使い学習を行っています．日本語でも使用することができます．

LLaVAのv1.5のモデルは以下のものが現在利用できます．
- [llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- [llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b)

OSSの謎のマスコットは"a cute lava llama with glasses"というプロンプトで生成したそうです．
![スクリーンショット 2023-10-11 5.10.21.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/237bc6af-5736-bcb3-cec5-e895452166bc.png)



# 2. 使い方

すぐに試したい方はData Science WikiのページまたはColabのリンクから実行してみてください

<a href="https://www.data-science-wiki.net/article?path=/multimodal/image_to_text/llava.html">
<img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>

<a href="https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/develop/multimodal/image_to_text/llava.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

今回の検証では4bit量子化を行ったので，それぞれGPUの消費は以下でした．
llava-v1.5-7b：7.3 / 40.0 GB
llava-v1.5-13b：10.8 / 40.0 GB

今回はllava-v1.5-13bで検証を行っていきます．

必要なものをインストールします．

```python
!git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
!pip install -e .
```

```python
from PIL import Image
import requests
```

はじめに以下の画像で試してみます．

```python
url = "https://img.peapix.com/e27fcf12e0664a5cb1c6b58c6b311d31.jpg?attachment&modal"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image.save("/content/tokyo.jpg", "JPEG")
image
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/5359d13b-3f04-44d2-f1b3-26c3e723efdd.png)

以下のコマンドを実行するとLLaVAのダウンロードが始まります．その後にユーザーが入力できるコンソールウィンドウが表示されます．

```python
!python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-13b \
    --image-file "/content/tokyo.jpg" \
    --load-4bit
```

出力結果
![スクリーンショット 2023-10-11 4.18.40.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/c3b2dcba-ea93-693d-62e5-0a748d2e5fca.png)
大体，正しい出力を得ることができていますが東京タワーについては回答できていません．この辺りはインストラクションデータに含まれていないと難しいのかもしれません．

次に以下の画像で試します．

```python
url = "https://i.gzn.jp/img/2013/12/08/cat-island-aoshima/P1640392.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image.save("/content/cat.jpg", "JPEG")
image
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/e603d245-b2aa-0c1d-7458-800910fb7d69.png)


```python
!python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-13b \
    --image-file "/content/cat.jpg" \
    --load-4bit
```

出力結果
![スクリーンショット 2023-10-11 4.32.49.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/ecdfb7ae-b525-68b4-3807-8e674a0a84a4.png)
正確には猫の数は当たっていませんが，ある程度は猫のことを認識している印象です．また，最後の質問は答えられるとは思いませんでしたが，そこまで正確ではないもののまともな回答ができています．このレベルのモデルがOSSとして使えるのは非常にありがたいですね．

# 4. おわりに

今回はLLaVAの紹介でした．前に紹介した[BLIP,BLIP2](https://qiita.com/fuyu_quant/items/50e0b2a9a0b5947527c6)に比べ画像に対する説明能力が飛躍的に上昇している印象です．このレベルで画像を文章に変換できるのであれば何かしら社会実装で役に立ちそうですね．

最後までお読みいただきありがとうございます．記事に誤り等ありましたらご指摘いただけますと幸いです。

# 5. 参考文献
- https://note.com/npaka/n/n3656c32f71af
