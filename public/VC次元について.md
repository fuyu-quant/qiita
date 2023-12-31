---
title: VC次元について
tags:
  - 機械学習
private: false
updated_at: '2021-04-14T17:51:53+09:00'
id: f94c51a00630deff2a80
organization_url_name: null
slide: false
ignorePublish: false
---
#初めに
機械学習アルゴリズムの説明に出てくるVC次元について説明します。
訂正箇所がありましたらご指摘くださいお願いします。

#目次
- [定義](#定義)
- [用語の確認](#用語の確認)
   - [入力空間](####入力空間)
   - [仮説](####仮説)
   - [仮説集合](####仮説集合)
   - [サンプル数](####サンプル数)
   - [二値分類](####二値判別問題)
   - [パラメータ](####パラメータ)
- [つまりどういうこと?](#つまりどういうこと?)  
- [別の視点で考える](#別の視点で考える)
- [VC次元を使った議論](#VC次元を使った議論)
   - [1,汎化の難しさについて](###1-汎化の難しさについて) 
   - [2,機械学習アルゴリズムの計算量について](###2-機械学習アルゴリズムの計算量について)
- [おわりに](#おわりに)
- [参考・引用文献](#参考・引用文献)


#定義
VC次元は主に、「二値判別問題のための仮説集合」に対して定義される複雑度のこと
定義は以下のようになります....

```math
\operatorname{VCdim}(\mathcal{H})=\max \left\{n \in \mathbb{N} \mid \max _{x_{1}, \ldots, x_{n} \in \mathcal{X}} \Pi_{\mathcal{H}}\left(x_{1}, \ldots, x_{n}\right)=2^{n}\right\}
```

式の意味としては、仮説集合$\mathcal{H}$の仮説$h$による$(x_1,...,x_n)$のラベルのセット$(h(x_1),...h(x_n))$の数を$=2^n$とした時の最大のサンプル数$n$のこと。
ちょっとよく分からないですね...

#用語の確認
まず初めに簡単に用語を説明します。

####入力空間
$\mathcal{X}$
入力$(x_1,...x_n)$の集合、訓練データ$((x_1,y_1),...,(x_n,y_n))$の正解ラベル$y$ではないほう

####出力ラベル
$(y_1,...y_n)$
各入力に対する、その正解ラベル

####仮説
$h(x)$
機械学習アルゴリズムが生成するもの、つまり学習済みのモデルそのもののこと。
入力空間から出力集合への関数

####仮説集合
$\mathcal{H}$
仮説集合が複数集まったもの、ある機械学習アルゴリズムによって生成される可能性のあるモデルの集まり

####サンプル数
$n$
機械学習アルゴリズムが訓練に用いるデータの数

####二値判別問題
その名の通り、出力結果が0か１でどちらかに分類する問題

####パラメータ
機械学習アルゴリズが持つ変数で、学習により値が決まる。
訓練データにより推定する対象。
(一方、ハイパーパラメータはパラメータの推定を支援するプロセスで使用されるもので訓練データから値を推定できない)

#つまりどういうこと?
ある仮説集合$\mathcal{H}$があったとする。
仮説集合にはいくつかの仮説が含まれている。
![スクリーンショット 2021-04-11 3.39.28（2）.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/66bc817e-14c9-1482-9152-316659040226.png)
例えば、仮説$h_1(x)$について考える。
$(x_1,...x_n)$とその正解ラベル$(y_1,...,y_n)$がある。$h_1(x)$に対してサンプル数$n$の入力$(x_1,...x_n)$があった時、出力とし$(h_1(x_1),...h_1(x_n))$を得る
二値分類であるため全ての$h_1(x_i)$($i\in $\{1,2,...n\}$)$は0か１のどちらかとなり、$(0,1,1,1,0,...1,0,0,0)$などのような出力となる。
一方、サンプル数が$n$個あった時の考えられる分類の方法は$2^n$通りであることがわかる。
もし仮説集合の中に出力結果が異なる仮説$h(x)$が$2^n$種類(同じ出力となるものは同じ仮説とみなす)存在する場合、その中のどれかを使うことで入力$(x_1,...x_n)$に対する出力ラベルと完全に一致する仮説$h(x)$が必ず存在するはずである。
ここで、仮説集合$\mathcal{H}$の中の仮説$h(x)$の数を

```math
\Pi_{\mathcal{H}}(x_{1}, \ldots, x_{n})

```
と表すとする。
サンプル数が増えると分類パターンが増えていく、この数が仮説集合の中に含まれている仮説$h(x)$の数より増えた場合は

```math
\Pi_{\mathcal{H}}\left(x_{1}, \ldots, x_{n}\right) < 2^{n}
```
となり、完全に分類することができない。
つまりVC次元とは、ある仮説集合の中に含まれている仮説によって完璧に分類することができる状態、

```math
\Pi_\mathcal{H}(x_{1}, \ldots, x_{n}) = 2^{n}
```
が成り立つ最大の$n$のことである。

#別の視点で考える
定義通りに説明をすると上記のような話になるが、今度は機械学習のパラメータに紐付けて説明することを考える。

仮説集合$\mathcal{H}$と$\mathcal{H}$の要素を増やした仮説集合$\mathcal{H}'$があるとする。

![スクリーンショット 2021-04-11 16.43.34.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/5c59d249-26ce-7883-39e8-ccb3ffd38916.png)

この仮説集合$\mathcal{H}$と$\mathcal{H}'$とでは分類できるバリエーションが$\mathcal{H}'$の方が多い。
仮説の一つ一つが異なるラベル付けを生成するからだ。
つまり、仮説が多い仮説集合ほど分類できるバリエーションが豊富であることがわかる。
では仮説集合の仮説の多さは何に起因しているのでしょうか?

それは、仮説を生成する機械学習アルゴリズムに含まれるパラメータの数です。

パラメータの数が多いほど、機械学習アルゴリズム内の処理が複雑になり「条件による分岐」や「重み付け」の多様性を生み出すことができます。

つまり、
![スクリーンショット 2021-04-11 17.19.56.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/c785d0c0-3ead-d8be-7399-2c0fbcb50ce4.png)


ということが言える。
機械学習アルゴリズムのパラメータを増やすことで、表現力が高くなりその結果VC次元も大きくするのだ。


#VC次元を使った議論
VC次元は実際にどのような議論で出てくるのでしょうか

###1 汎化の難しさについて
汎化誤差と経験誤差について、以下の不等式が成り立つことが知られている。

```math
(\text { 汎化誤差 }) \leq(\text { 経験誤差})+\mathcal{O}\left(\sqrt{\frac{\log \left(M / \text{VC次元}\right)}{M / \text{VC次元}}}\right)
```
これによると、例えばVC次元が非常に大きくなった時は右辺も非常に大きい値となってしまい汎化誤差を小さい値で抑えられなくなってしまうことが分かる。

###2 機械学習アルゴリズムの計算量について
機械学習アルゴリズムの計算量の評価にはしばしばVC次元が用いられる。
つまり、VC次元があまりにも大きかったり、VC次元に対して指数的なオーダーが含まれているとあまりよりアルゴリズムではないことがわかる。

詳しくはいつか書きます。

#おわりに

間違っている表現などがあると思いますが、誰かしらの役に立てたら幸いです。

#参考・引用文献
- 統計的学習理論(機械学習プロフェッショナルシリーズ)
- ディープラーニングと物理学 原理がわかる、応用ができる(KS物理専門書)
- ブースティング　ー学習アルゴリズムの設計技法ー
