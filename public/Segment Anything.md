---
title: Segment Anything
tags:
  - CV
  - segmentation
  - Segmentanything
private: false
updated_at: '2023-10-11T03:36:16+09:00'
id: e27505c00e92aa2a1acd
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

今回はSegment Anythingの紹介になります．Segment AnythingはMetaが発表した大量の画像データで学習したセグメンテーションモデルです．zero-shot-segmentationが可能であり学習データにない画像にも適用することができます．

記事に誤り等ありましたらご指摘いただけますと幸いです。

https://github.com/facebookresearch/segment-anything


# 目次
- [1. Segment Anything](#1-segment-anything)
- [2. 使い方](#2-使い方)
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)

# 1. Segment Anything

ライセンス:Apache-2.0 (研究目的限定)
リポジトリ:https://github.com/facebookresearch/segment-anything
公式サイト:https://segment-anything.com/
論文:https://arxiv.org/abs/2304.02643

Metaが発表したセグメンテーションモデルで，1100万枚のライセンス画像とプライバシーを尊重した画像と、110 万枚の高品質セグメンテーションマスクデータ、10億以上のマスクアノーテションという過去最大のデータセットで訓練されたモデルです．特に、zero-shot-segmentationが可能でかつ精度が教師あり学習を超える精度を出すこともできるそうです．

https://github.com/facebookresearch/segment-anything

# 2. 使い方

すぐに試したい方はData Science WikiのページまたはColabのリンクから実行してみてください

<a href="https://www.data-science-wiki.net/article?path=/cv/semantic_segmentation/segment_anything.html"><img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>

<a href="https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/develop/cv/semantic_segmentation/segment_anything.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

必要なライブラリのインストール

```python
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install pycocotools onnxruntime onnx supervision
```

モデルのダウンロード

他にも3種類のチェックポイントのモデルが用意されています．
(https://github.com/facebookresearch/segment-anything#model-checkpoints)

- `default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

```python
# モデルのダウンロード
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

適当な画像をダウンロード

```python
!wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg
```

```python
import numpy
import torch
import matplotlib.pyplot as plt
import cv2
import supervision as sv

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
```

元の画像の表示します．

```python
IMAGE_PATH = "/content/dog.jpeg"

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Annotated Image")
plt.axis("off")  # 軸をオフにして、画像だけを表示します。
plt.show()
```
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/ad392a6d-ccaa-14ef-2b1e-26a48ecd580b.png)

ダウロードしたモデルを読み込みハイパーパラメータの設定を行います．
ハイパーパラメータについては[このサイト](https://www.chowagiken.co.jp/blog/sam_parameter)が参考になります．

```python
sam = sam_model_registry["vit_h"](checkpoint="/content/sam_vit_h_4b8939.pth")

# モデルをGPUに載せる
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam.to(device=DEVICE)

# パラメータの設定
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side = 32,
    pred_iou_thresh = 0.980,
    stability_score_thresh = 0.96,
    crop_n_layers = 1,
    crop_n_points_downscale_factor = 2,
    min_mask_region_area = 100
)
```
セグメンテーションの実行
```python
result = mask_generator.generate(image_rgb)
mask_annotator = sv.MaskAnnotator(color_map = "index")
detections = sv.Detections.from_sam(result)
annotated_image = mask_annotator.annotate(image_bgr, detections)
```
セグメンテーション結果の表示
```python
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.imshow(annotated_image_rgb)
plt.title("Annotated Image")
plt.axis("off")
plt.show()
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/45b4451b-f85d-e271-a7e3-853e01022955.png)

また出力結果には以下のような情報も含まれています．

```python
result

# 出力
[{'segmentation': array([[False, False, False, ...,  True, False, False],
         [False, False, False, ...,  True, False, False],
         [False, False, False, ...,  True, False, False],
         ...,
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False],
         [False, False, False, ..., False, False, False]]),
  'area': 234889,
  'bbox': [3, 0, 714, 552],
  'predicted_iou': 1.0237175226211548,
  'point_coords': [[168.75, 100.0]],
  'stability_score': 0.9896397590637207,
  'crop_box': [0, 0, 720, 1280]},
```

それぞれ

- segmentation:(W, H)の形状を持つマスクでbool型
- area:マスクのピクセルの面積
- bbox:xywhフォーマットのマスクのボックス
- predicted_iou:マスクの品質に関するモデル自身の予測値
- point_coords:このマスクを生成したサンプリングされた入力点
- stability_score:マスクの品質に関する追加的な指標(詳細はよくわかりません)
- crop_box:このマスクを生成するために使用された画像のクロップ（xywh形式）

# 3. おわりに

今回はAutomatic Prompt Engineerの紹介でした．

記事に誤り等ありましたらご指摘いただけますと幸いです。

# 4. 参考文献
- https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
- https://www.chowagiken.co.jp/blog/sam_parameter
