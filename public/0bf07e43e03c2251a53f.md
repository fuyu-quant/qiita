---
title: VALL-E X
tags:
  - MultiModal
  - VALL-E-X
private: false
updated_at: '2023-10-05T19:16:17+09:00'
id: 0bf07e43e03c2251a53f
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに

こんにちは、[fuyu-quant](https://twitter.com/fuyu_quant)です．

今回はVALL-E Xの紹介になります．英語，中国語，日本語の３つ言語での音声合成，感情やアクセントを考慮した音声合成，言語をまたいだ音声合成などを行うことができます．さらに自分が用意した数秒の音声ファイルと文字起こししたものを与えることで音声を複製することもできます．
記事に誤り等ありましたらご指摘いただけますと幸いです。

https://github.com/Plachtaa/VALL-E-X

# 目次
- [1. VALL-E X](#1-vall-e-x)
- [2. 使い方](#2-使い方)
    - [音声合成(英語)](#音声合成(英語))
    - [音声合成(日本語)](#音声合成(日本語))
    - [音声合成(複数言語(英語-中国語))](#音声合成)
    - [様々な音声の選択](#様々な音声の選択)
    - [Voice Cloning(音声複製)](#voice-cloning(音声複製))
- [3. おわりに](#3-おわりに)
- [4. 参考文献](#4-参考文献)

# 1. VALL-E X

ライセンス:MIT(商用利用可能)
リポジトリ:https://github.com/Plachtaa/VALL-E-X
論文:https://arxiv.org/abs/2303.03926

VALL-E XはMicrosoftが開発した音声合成のモデルです。論文はMicrosoftから出ていますが、今回紹介するモデルはMicrosoftaが出しているモデルではないです(Microsoftの公式のモデルは公開されていないみたいです)主な特徴としては、英語、日本語、中国語での音声合成と数秒の音声ファイルからその声を模倣した合成音声を生成できます。
アーキテクチャなどについては以下のサイトを参考にしてみてください。
[VALL-E-X : 再学習不要で声質を変更できる音声合成モデル](https://medium.com/axinc/vall-e-x-%E5%86%8D%E5%AD%A6%E7%BF%92%E4%B8%8D%E8%A6%81%E3%81%A7%E5%A3%B0%E8%B3%AA%E3%82%92%E5%A4%89%E6%9B%B4%E3%81%A7%E3%81%8D%E3%82%8B%E9%9F%B3%E5%A3%B0%E5%90%88%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB-977efc19ac84)

# 2. 使い方

音声ファイルの埋め込み方が分からなかったので，実際に生成した音声を確認したい人は以下のリンクで実行してみてください．

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fuyu-quant/data-science-wiki/blob/main/multimodal/VALL-E_X.ipynb)

以下のコマンドでインストールします．

```python
!git clone https://github.com/Plachtaa/VALL-E-X.git
cd VALL-E-X
!pip install -r requirements.txt
```

```python
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
```

モデルのダウンロード

```python
preload_models()
```

## 音声合成(英語)

英語での音声合成は長い文章でもクオリティの高い音声を作成することができました．

```python
text_prompt = """
    Quantum computers are advanced computational devices that use principles of quantum mechanics to process information. Unlike classical computers which use bits as 0s or 1s, quantum computers use qubits, which can be in a superposition of both states. This allows them to solve certain problems much faster than classical computers, especially in areas like cryptography and optimization.
"""
audio_array = generate_audio(text_prompt)

write_wav("/content/english.wav", SAMPLE_RATE, audio_array)
Audio(audio_array, rate=SAMPLE_RATE)
```

## 音声合成(日本語)

日本語は短い言葉であれば非常に上手く音声合成をすることができますが，長い文章の時はどんどんホラーのような音声になってしまいます．

```python
text_prompt = """
    ひき肉でーーーす．
"""
audio_array = generate_audio(text_prompt, prompt="cafe")

write_wav("/content/japanese.wav", SAMPLE_RATE, audio_array)
Audio(audio_array, rate=SAMPLE_RATE)
```

## 音声合成(複数言語(英語-中国語))

```python
text_prompt = """
    [EN]Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.[EN]
    [ZH]神经网络（Neural Network）是模拟人脑工作机制的算法，用于识别模式和处理数据。它包括多个层，每个层都有许多神经元。通过训练数据，神经网络可以不断调整其内部权重，以优化其预测和分类能力。[ZH]
"""
audio_array = generate_audio(text_prompt, language='mix')

write_wav("/content/multilingual.wav", SAMPLE_RATE, audio_array)
Audio(audio_array, rate=SAMPLE_RATE)
```

## 様々な音声の選択

以下のリンクから

 [https://github.com/Plachtaa/VALL-E-X/tree/master/presets](https://github.com/Plachtaa/VALL-E-X/tree/master/presets)

いくつか試しましたが選択する音声ごとに聞き取りやすい言語は異なりました．

またさまざまな感情の音声も用意されています．

```python
text_prompt = """
I'll get serious starting tomorrow
"""

audio_array = generate_audio(text_prompt, prompt="cafe")

write_wav("/content/sample.wav", SAMPLE_RATE, audio_array)
Audio(audio_array, rate=SAMPLE_RATE)
```

## Voice Cloning(音声複製)

自分が手元で用意した音声のデータを使い，音声を複製する方法です．

今回は音声合成したものを利用し，同じ声で音声合成しています

```python
from utils.prompt_making import make_prompt

# 先ほど生成した音声を使う
make_prompt(name="sample", audio_prompt_path="/content/sample.wav",
                transcript="I'll get serious starting tomorrow.")

# whisperが使える時は文章を明示的に与えなくても自動で文字起こしができる
#make_prompt(name="paimon", audio_prompt_path="paimon_prompt.wav")

text_prompt = """
Procrastination is the thief of time
"""
audio_array = generate_audio(text_prompt, prompt="sample")

write_wav("/content/sample2.wav", SAMPLE_RATE, audio_array)
Audio(audio_array, rate=SAMPLE_RATE)
```

# 3. おわりに
今回は音声合成モデルのVALL-E Xの紹介でした。英語、日本語、中国語で利用することができ、特に英語では長い文章であっても綺麗な音声で出力することができました。またVoice Cloning(音声複製)では数秒のデータを与えるだけですぐに複製ができました。
記事に誤り等ありましたらご指摘いただけますと幸いです。

# 4. 参考文献
- https://github.com/Plachtaa/VALL-E-X
- [VALL-E-X : 再学習不要で声質を変更できる音声合成モデル](https://medium.com/axinc/vall-e-x-%E5%86%8D%E5%AD%A6%E7%BF%92%E4%B8%8D%E8%A6%81%E3%81%A7%E5%A3%B0%E8%B3%AA%E3%82%92%E5%A4%89%E6%9B%B4%E3%81%A7%E3%81%8D%E3%82%8B%E9%9F%B3%E5%A3%B0%E5%90%88%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB-977efc19ac84)
