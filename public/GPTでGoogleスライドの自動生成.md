---
title: GPTでGoogleスライドの自動生成
tags:
  - slide
  - gpt-2
  - gpt-3
private: false
updated_at: '2023-02-23T15:41:24+09:00'
id: b9b403a5a5bd57d2472e
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに
この記事はGoogle slideでスライドの自動生成を行うための設定方法の紹介になります。GPT for Slidesというものを利用してGoogle slideで好きなテーマのスライド生成を行います。以下が今回作成した資料です。
<script defer class="speakerdeck-embed" data-slide="1" data-id="1be2736c45a34a339afe07001b0e13e5" data-ratio="1.77725118483412" src="//speakerdeck.com/assets/embed.js"></script>

# 目次
- [1. アドオンのインストール](#1.-アドオンのインストール)
- [2. OpenAIのアカウント作成とAPIキーの取得](#2.-OpenAIのアカウント作成とAPIキーの取得)
- [3. MagicSlidesへのAPIキーの設定](#3.-GoogleSlideへの設定)
- [4. スライドの自動生成](#4.-スライドの自動生成)
- [5. 生成資料(再掲)](#5-参考文献)
- [6. 現状](6.-現状)
- [7. 参考文献](#7.-参考文献)


# 1. アドオンのインストール
以下のサイトから「MagicSlides App」をインストールします
* [MagicSlides App](https://workspace.google.com/marketplace/app/magicslides_app/371894645570)
![スクリーンショット 2023-02-23 14.37.47.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/a222af97-1a88-c7e8-822b-25636bc33f12.png)
* 「続行」を選択
![スクリーンショット 2023-02-23 14.39.55.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/a91effb6-f056-1af2-dde8-e0503618f4e1.png)
* Googleアカウントを一つ選択する
![スクリーンショット 2023-02-23 14.40.38.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/f0cdbbfe-adca-7009-5aec-f6f89fae871c.png)

* 「MagicSlides App」について以下を許可する
![スクリーンショット 2023-02-23 14.42.54.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/4df81df1-5ae5-23fa-f86c-38a175bc3613.png)
* [MagicSlides App]の取得はこれで完了

# 2. OpenAIのアカウント作成とAPIキーの取得
* [OpenAI](https://auth0.openai.com/u/signup/identifier?state=hKFo2SBlajRxWVY1cEt3QWVPaW9FQ1hpenkwVHZ1N0JHQ1FfSaFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIG1EWHJzZ1lySHdvRGZZMEpYRXlmRmdEeElGSUZiTVU2o2NpZNkgRFJpdnNubTJNdTQyVDNLT3BxZHR3QjNOWXZpSFl6d0Q)でアカウントを作成
![スクリーンショット 2023-02-23 14.51.49.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/e7a1f638-506a-dce5-41d6-d2c0fb5b6430.png)
* アカウントを作成後に[ここのページ](https://platform.openai.com/account/api-keys)の「Create new secret key」を作成
![スクリーンショット 2023-02-23 14.53.57.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/e2d29c44-679d-4f92-d0d2-db32dfc3b42c.png)
* 作成した「API key」をコピーする
![スクリーンショット 2023-02-23 14.55.12.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/8180c5d9-7708-372a-2235-e726de899de1.png)


# 3. GoogleSlideへの設定
MagicSlidesへのAPIキーの設定を行う
* 新しくスライドを作成する
* 「拡張機能」を選択し、「MagicSlides App」を選択する
![スクリーンショット 2023-02-23 15.03.11.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/6189037b-b044-16b7-87eb-992b8e0457d9.png)
* 以下の青で囲われた箇所に先ほどコピーしたAPIキーを貼り付け、「Save」を押す
![スクリーンショット 2023-02-23 15.05.17.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/ad08646c-165f-9ee4-d6de-16030f95da7b.png)

# 4. スライドの自動生成
以上で設定が完了したのでスライドの自動生成を行う
今回は「量子コンピュータ」のスライドを「専門家」向けに7枚作ってもらう。
* 以下のように入力する
    * Topic：「Quantum Computer」
    * Total number of slides:7（実際には表紙と背表紙も作られるの9枚になる）
    * Add Info (optional)：professional
![スクリーンショット 2023-02-23 15.10.37.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/49736f3f-adcc-3666-43b1-3025e2a1b62b.png)

# 5. 生成資料(再掲)
以下が今回作成した、「量子コンピュータ」の「professional」に関する情報を追加した資料です
<script defer class="speakerdeck-embed" data-slide="1" data-id="1be2736c45a34a339afe07001b0e13e5" data-ratio="1.77725118483412" src="//speakerdeck.com/assets/embed.js"></script>

# 6. 現状
* 資料としてはまだまだですが、今後に期待が持てるような機能だと思いました
* スライドで大切なグラフや図の挿入は「Add Info」で色々試しましたができませんでした(もしやり方があれば教えてくだい!)
* 内容については厳密には違うこともありますが大体のことは正しいような気がします
* 追加のつけた「professional」については、8枚目のスライドの「Professional Use」で反映されているみたいです
* 今後は画像生成の時のようにスライドでもプロンプトエンジニアリングのような話が出てくるかもしれませんね〜


# 7. 参考文献
* https://qiita.com/Qiita/items/612e2e149b9f9451c144
* https://www.gptforslides.app
