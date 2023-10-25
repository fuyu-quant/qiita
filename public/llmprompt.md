---
title: LLMのプロンプト技術まとめ
tags:
  - プロンプト
  - ChatGPT
  - LLM
  - GPT-4
  - prompt_engineering
private: false
updated_at: '2023-10-25T17:41:22+09:00'
id: 157086987bd1b4e52e80
organization_url_name: null
slide: false
ignorePublish: false
---
# はじめに
今回はすぐに使えそうなプロンプトの工夫や生成方法について有名なものをまとめました．LMの精度向上のの役立てられればと思います．

※記事に誤り等ありましたらご指摘いただけますと幸いです．
※記載されている順番は論文が発表されている順番とは関係ありません．
※論文で精度向上が確認されているのは英語での検証のみで，日本語では改善されるか分かりません．
※全てのLLMで精度が改善するとは限りません．



# 目次
- [Zero-shot prompting](#zero-shot-prompting)
- [Few-shot prompting](#few-shot-prompting)
- [Chain of Thoughts(CoT)](#chain-of-thoughtscot)
- [Few-shot Chain of Thoughts](#few-shot-chain-of-thoughts)
- [Zero-shot Chain of Thoughts (step by step)](#zero-shot-chain-of-thoughts-step-by-step)
- [Tree of Thoughts(ToT)](#tree-of-thoughtstot)
- [Algorithm of Thoughts(AoT)](#algorithm-of-thoughtsaot)
- [Graph of Thoughts(GoT)](#graph-of-thoughtsgot)
- [Self-translate](#self-translate)
- [Take a Deep Breath](#take-a-deep-breath)
- [Take a Step Back](#take-a-step-back)
- [Automatic Prompt Engineer](#automatic-prompt-engineer)
- [Chain-of-Verification(CoVe)](#chain-of-verificationcove)
- [Metacognitive Prompting(MP)](#metacognitive-promptingmp)
- [Self-Consistency](#self-consistency)
- [Generated Knowledge Prompting](#generated-knowledge-prompting)
- [EchoPrompt](#echoprompt)
- [Chain of density(CoD)](#chain-of-densitycod)
- [その他](#その他)
- [おわりに](#おわりに)


# Zero-shot prompting
大量の学習データを使ったLLMは例などを示さずにある程度の質問には回答することができます．Zero shot promptingは手法というよりかはLLMに対してただ聞くだけのプロンプトで，他の手法と比較する際のベースとして使うことが多いです．

- 具体例
    ```python
    prompt = """
    ロジャーはテニスボールを5個持っている。彼はさらに2つのテニスボール缶を買った。それぞれの缶には3個のテニスボールが入っている。彼は今何個のテニスボールを持っていますか？
    """
    ```



# Few-shot prompting
In-context learningを可能にするプロンプティング手法．いくつかの例を示すことで類似の質問に対する回答の精度を上げる

- 具体例
    ```python
    # 以下はOne shot prompting
    prompt = """
    Q：ロジャーはテニスボールを5個持っている。彼はさらに2つのテニスボール缶を買った。それぞれの缶には3個のテニスボールが入っている。彼は今何個のテニスボールを持っていますか？

    A: 11個

    Q: 食堂には23個のリンゴがあった。昼食に20個使い、さらに6個買ったとすると、りんごは何個あるか。
    """
    ```

- 参考文献
    - [https://www.promptingguide.ai/techniques/fewshot](https://www.promptingguide.ai/techniques/fewshot)





# Chain of Thoughts(CoT)
論文:[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
思考の過程をプロンプトに入力することでLLMに複雑な推論を可能にするプロンプト手法．

- 具体例
    以下は通常のプロンプトとChain of Thoughtsのプロンプトになります．
    ```python
    # Few shot prompting
    prompt = """
    Q：ロジャーはテニスボールを5個持っている。彼はさらに2つのテニスボール缶を買った。それぞれの缶には3個のテニスボールが入っている。彼は今何個のテニスボールを持っていますか？

    A: 11個

    Q: 食堂には23個のリンゴがあった。昼食に20個使い、さらに6個買ったとすると、りんごは何個あるか。
    """

    # Chain of Thoughts
    prompt = """
    Q：ロジャーはテニスボールを5個持っている。彼はさらに2つのテニスボール缶を買った。それぞれの缶には3個のテニスボールが入っている。彼は今何個のテニスボールを持っていますか？

    A: ロジャーは5個のテニスボールをはじめに持っていました。テニスボール3個入りの缶を2つ買うと、テニスボール6個になります。5 + 6 = 11. 答えは11です。

    Q: 食堂には23個のリンゴがあった。昼食に20個使い、さらに6個買ったとすると、りんごは何個あるか。
    """
    ```

- 参考記事
    - [https://qiita.com/shiro-manju/items/470132e29b1ab984a4d1](https://qiita.com/shiro-manju/items/470132e29b1ab984a4d1)



# Few-shot Chain of Thoughts
Chain of Thoughtsでプロンプトに与える思考の過程を複数用意する手法



# Zero-shot Chain of Thoughts (step by step)
論文:[Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
プロンプトに「Let’s think step by step」を追加するだけでZero shot promptingを大きく改善することができる手法．

- 具体例

    ```python
    英語
    prompt = """
    答えたい問題
    Let's think step by step
    """

    # 日本語
    prompt = """
    答えたい問題
    段階的に考えてください．
    """
    ```

    その他の論文で検証されたstep by step関連のプロンプトのまとめは以下になります．

![スクリーンショット 2023-10-25 16.59.14.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/a5f34811-f844-c61c-6a9f-cca01703bb1f.png)


- 参考記事
    - https://blog.shikoan.com/zero-shot-cot/



# Tree of Thoughts(ToT)
論文:[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
Chain of Thoughtsとツリー探索を組み合わせたような手法．
使い方は以下にまとまっています．
(※以下のOSSでは特定の問題を解くためのものしか公開されていません)

https://qiita.com/fuyu_quant/items/c329d3fa80bda5efd32a



# Algorithm of Thoughts(AoT)
論文:[Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models](https://arxiv.org/abs/2308.10379)
使い方は以下にまとまっています．
(※以下のOSSでは特定の問題を解くためのものしか公開されていません)
<a href="https://www.data-science-wiki.net/article?path=/nlp/llm_framework/algorithm_of_thoughts.html"><img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>



# Graph of Thoughts(GoT)
論文:[Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Large Language Models](https://arxiv.org/abs/2305.16582)
使い方は以下にまとまっています．
(※以下のOSSでは特定の問題を解くためのものしか公開されていません)
<a href="https://www.data-science-wiki.net/article?path=/nlp/llm_framework/graph_of_thoughts.html"><img src="https://raw.githubusercontent.com/fuyu-quant/data-science-wiki/main/images/logo2.png" alt="Data Science Wiki" width="400"/>
</a>



# Self-translate
論文:[Do Multilingual Language Models Think Better in English?](https://arxiv.org/abs/2308.01223)
英語以外の言語のプロンプトでは言語モデルはその能力をフルに活用できていない，という研究に関する論文．self-translateは入力を一度英語に翻訳して出力を得てから元の言語に戻す手法．

- 具体例

    以下では日本語の入力に対してOne shotで英語に翻訳し，それをOne shot promptingで回答している．

    ```python
    # 従来のプロンプト
    prompt = """
    アンディはゼラニウムを90本植え、ペチュニアをゼラニウムより40本少なく植える。全部で何本の花を植えたでしょう？
    """

    # Self-translate
    prompt1 = """
    Japanese:メアリは緑の魔女を叩かなかった．
    English:Mary did not slap the green witch.

    Japanese:アンディはゼラニウムを90本植え、ペチュニアをゼラニウムより40本少なく植える。全部で何本の花を植えたでしょう？
    English:
    """

    # prompt1の出力
    Andy plants 90 geraniums and 40 less petunias than geraniums. How many flowers does he plant in total?

    prompt2 = """
    Question: John had 12 apples and ate a quarter of them. How many apples does he have now?
    Stepwise solution: John ate 12÷4=3 apples. Therefore, he has 12–3=9 apples now. The answer is 9.

    Question: Andy plants 90 geraniums and 40 less petunias than geraniums. How many flowers does he plant in total?
    Stepwise solution:
    """

    # prompt2の出力
    Andy plants 90 geraniums and 90-40=50 petunias. Therefore, he plants 90+50=140 flowers. The answer is 140.
    ```

- 参考文献
    - [https://github.com/juletx/self-translate/](https://github.com/juletx/self-translate/)


# Take a Deep Breath
論文:[Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
「Take a deep breath」とプロンプトに加えることで精度が改善する手法．

- 具体例
    ```python
    英語
    prompt = """
    Take a deep breath and
    質問
    """

    # 日本語
    prompt = """
    深呼吸をしてから答えてください
    質問
    """
    ```




# Take a Step Back
論文:[Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117)
LLMに推論するための概念や原理を与えることで正しく推論するための精度を向上させる手法．

- 具体例

    ```python
    # 通常のプロンプト
    prompt = """
    温度が2倍、体積が8倍になると、理想気体の圧力Pはどうなりますか？
    """

    # Take a Step Back
    prompt1 = """
    「温度が2倍、体積が8倍になると、理想気体の圧力Pはどうなりますか？」
    --------
    上記の質問の背景にある物理学の原理は何ですか？
    """
    ## prompt1の出力
    理想気体の法則： ここで、Pは圧力、Vは体積、nはモル数、Rは気体定数、Tは温度である。

    prompt2 = """
    温度が2倍、体積が8倍になると、理想気体の圧力Pはどうなりますか？

    理想気体の法則： ここで、Pは圧力、Vは体積、nはモル数、Rは気体定数、Tは温度である。
    """

    # prompt2の出力
    理想気体の法則を適用して圧力を求める。温度が2倍になれば、Tは2Tになる。体積が8倍になれば、Vは8Vになる。これらの値を理想気体の法則に代入すると、こうなる： P(8V) = nR(2T) 両辺を8Vで割ると、こうなる： P = nR(2T) / 8V 圧力が4分の1になったことがわかる。
    ```

- 参考文献
    - https://aiboom.net/archives/56671
    - https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb



# Automatic Prompt Engineer
論文:[Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910)
入力と出力のペアをいくつか与えると，その入力から出力を得るための精度のよいプロンプトを自動的に見つけてくれます．さまざまな応用ができそうです．
使い方については以下にまとめています．

https://qiita.com/fuyu_quant/items/d90c7d69b143d59a7b84




# Chain-of-Verification(CoVe)
論文:[Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495)
LLMの出力からハルシネーション(事実と異なる間違い)を減らすための手法です．
以下のようなステップで検証を行います．

1. ある入力に対するLLMから回答を得る(検証したい文章)
2. 検証するための質問文をいくつか生成する
3. それぞれの質問に対して独立して回答する
4. 最終的な検証済みの回答を生成
- 具体例

    ```python
    # 1のプロンプト(※Few-shot　promptingを使っている)
    prompt1 = """
    Q: 質問1
    A: 1の答え
    Q: 質問2
    A: 2の答え
    Q: 質問3
    A:
    """
    # 出力
    3の答え

    # 2のプロンプト(※Few-shot promptingを使っている)
    prompt2 = """
    Context: Q:質問1
    A:1の答え
    Response:
    1の答えに含まれている事実，それを検証するための質問
    1の答えに含まれている事実，それを検証するための質問

    Context: Q:質問2
    A:2の答え
    Response:
    2の答えに含まれている事実，それを検証するための質問
    2の答えに含まれている事実，それを検証するための質問

    Context: Q:質問3
    A:3の答え
    Response:
    """
    # 出力
    質問3の答えに含まれている事実とそれを検証するための質問がいくつか出力される

    # 3のプロンプト，検証する質問ごとに答えを生成する(※Few-shot promptingを使っている)
    prompt3 = """
    Q:検証するための質問
    A:答え

    Q:検証するための質問
    A:答え

    Q:先ほど生成した質問
    A:
    """

    # 検証用の質問に対する答え

    # 4のプロンプト(※Few-shot promptingを使っている)
    prompt4 = """
    Context:修正前の文章
    別のソースからの情報
    検証ステップの実行結果:Q + A
    検証ステップの実行結果:Q + A
    Response:修正された一貫性のある文章

    Context:修正前の文章
    別のソースからの情報
    検証ステップの実行結果:Q + A
    検証ステップの実行結果:Q + A
    Response:修正された一貫性のある文章

    Context:prompt1の答え
    別のソースからの情報
    検証ステップの実行結果:Q(prompt2の質問) + A(prompt3の答え)
    検証ステップの実行結果:Q(prompt2の質問) + A(prompt3の答え)
    Response:
    """
    ```

- 参考文献
    - [https://aiboom.net/archives/55711](https://aiboom.net/archives/55711)




# Metacognitive Prompting(MP)
論文:[Metacognitive Prompting Improves Understanding in Large Language Models](https://arxiv.org/pdf/2308.05342.pdf)
LLMに人間の認知プロセスを模倣させることで精度を改善した手法．

(作成途中)

- 具体例
    ここでは以下のステップでメタ認知を検証している
    1. 文章の理解
    2. 感情に関する予備的知識

    ```python
    # 通常のプロンプト
    prompt = """
    質問1: 「世界で最も美しいビーチは何ですか？」と質問2:「最も美しいビーチは何ですか？」、これら2つの質問がお互いの言い換えであるかどうかを判断してください。
    """

    prompt1 = """
    質問1: 「世界で最も美しいビーチは何ですか？」と質問2:「最も美しいビーチは何ですか？」、これら2つの質問がお互いの言い換えであるかどうかを判断してください。
    Q:両方の質問に対するあなたの理解を明確にしてください。
    A:
    """

    # 出力
    どちらの質問も、世界で最も美しいビーチについて尋ねているのだと理解している。

    prompt2 = """

    Q:両方の質問に対するあなたの理解を明確にしてください。
    A:どちらの質問も、世界で最も美しいビーチについて尋ねているのだと理解している。
    Q:主題、文脈、意味内容に基づく類似性の予備的識別を行う。
    A:
    """
    ```

- 参考文献
    - https://github.com/EternityYW/Metacognitive-Prompting/blob/main/prompts/five_shot_exemplars.pdf





# Self-Consistency
論文:[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171)
Few-shot CoTの出力結果の多数決を行い最終的な出力結果にする

- 具体例

    ```python
    # Few-shot examplerを使う(よりタスクに合わせた他のものでも良い)
    prompt = """
    Q：駐車場に車が3台あり、さらに2台の車が到着した場合、駐車場には何台の車がありますか？
    A：駐車場には既に3台の車があります。2台の車が到着しました。これで、車が3+2 = 5台あります。回答は5です。

    Q：リアは32個のチョコレートを持っており、彼女の姉妹は42個のチョコレートを持っています。彼らが35個食べた場合、彼らが残したピースの数は何ですか？
    A：リアは32個のチョコレートを持っており、リアの姉妹は42個のチョコレートを持っていたことを意味します。つまり、もともとは32 + 42 = 74個のチョコレートがありました。35個食べられました。したがって、合計で残るのは74-35 = 39個のチョコレートです。回答は39です。

    Q：ショーンは5つのおもちゃを持っています。クリスマスに、彼は両親からそれぞれ2つのおもちゃをもらいました。今、彼は何個のおもちゃを持っていますか？
    A：彼は5つのおもちゃを持っています。彼は母親から2つのおもちゃをもらいました。したがって、5 + 2 = 7個のおもちゃがあります。その後、父親から2つのおもちゃが追加されたので、合計で7 + 2 = 9個のおもちゃがあります。回答は9です。

    Q：マイケルは58個のゴルフボールを持っています。火曜日に、彼は23個のゴルフボールを失いました。水曜日に、さらに2個を失いました。水曜日の終わりには、彼は何個のゴルフボールを持っていましたか？
    A：マイケルは最初に58個のボールを持っていました。火曜日に23個を失いましたので、その後35個のボールが残りました。水曜日に2個を失ったので、現在33個のボールがあります。回答は33です。

    Q：オリビアは23ドル持っています。彼女は1つあたり3ドルのベーグルを5つ買いました。彼女が残したお金はいくらですか？
    A：彼女は1つあたり3ドルのベーグルを5つ購入しました。彼女は１５ドルを使った。残したお金は８ドルです。

    Q：私が6歳のとき、妹は私の半分の年齢でした。今、私は70歳です。私の妹は何歳ですか？
    A：

    """

    # 出力1
    私が6歳のとき、私の妹は私の半分の年齢であったため、彼女は3歳でした。今、私が70歳であるため、彼女は70-3 = 67歳です。回答は67です。

    # 出力2
    語り手が6歳のとき、彼の妹は彼の半分の年齢である3歳でした。語り手が70歳である今、彼の妹は70-3 = 67歳になるでしょう。回答は67です。

    # 出力3
    私が6歳のとき、私の妹は私の半分の年齢だったので、彼女は3歳でした。今、私は70歳なので、彼女は70/2=35歳です。答えは35です。

    上記の出力で多数派のものを最終的な出力結果とする。
    ```

- 参考文献
    - https://www.promptingguide.ai/jp/techniques/consistency
    - https://techblog.cccmk.co.jp/entry/2023/04/04/102443




# Generated Knowledge Prompting
論文:[Generated Knowledge Prompting for Commonsense Reasoning](https://arxiv.org/abs/2110.08387)
LLMに関連知識を生成させプロンプトに追加することで，いくつかのデータセットでSOTAを達成した手法です．

- 具体例

    ```python
    # 知識を生成するプロンプト
    prompt1 = """
    さまざまなものに関する数値的な事実を生成してください．
    例
    入力： ペンギンは<マスク>翼を持っている。
    知識： 鳥には2枚の翼がある。ペンギンは鳥の一種だ。

    入力： 平行四辺形の辺は<マスク>である。
    知識：長方形は平行四辺形である： 長方形は平行四辺形である。正方形は平行四辺形である。

    入力：庭には<mask>個の足がある。
    知識：1ヤードは3フィートである： 1ヤードは3フィートである。

    入力：水は<マスク>の状態で存在できる。
    知識：水は<mask>の状態で存在する： 物質の状態は固体、液体、気体である。

    入力：一般的な人間の手足は<マスク>である。
    知識：人間には2本の腕と2本の足がある： 人間には2本の腕と2本の足がある。

    入力:{答えたい問題}
    知識:
    """

    # 出力
    knowledge(「答えたい問題」に関する知識)

    # 生成した知識をプロンプトに追加する
    prompt2 = """
    答えたい問題
    knowledge
    """
    ```

- 参考文献
    - https://techblog.cccmk.co.jp/entry/2023/04/04/102443




# EchoPrompt
論文:[EchoPrompt: Instructing the Model to Rephrase Queries for Improved In-context Learning](https://arxiv.org/abs/2309.10687)
質問に答える前にその質問を言い換えるように指示をすることでいくつもの指標で精度が改善した手法．また，step by stepなどと組み合わせることもできる．

- 具体例

    ```python
    # 英語
    prompt = """
    Q:{答えたい問題}
    A:Let's repeat the question.
    """

    # 日本語
    prompt = """
    Q:{答えたい問題}
    A:質問を繰り返そう。
    """
    ```

- 参考文献
    - https://www.jbcc.co.jp/blog/column/innobase_202310_02.html




# Chain of density(CoD)
論文:[From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting](https://arxiv.org/abs/2309.04269)
テキストを要約する精度を高めるためのプロンプトです。

- 具体例

    ```python
    # 英語
    prompt = """
    Article:{要約したい文章}

    You will ask me for an article. Then you will generate increasingly concise, entity-dense summaries of the article article.

    Repeat the following 2 steps 5 times.

    Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.
    Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

    A missing entity is:
    - relevant to the main story,
    - specific yet concise (5 words or fewer),
    - novel (not in the previous summary),
    - faithful (present in the article),
    - anywhere (can be located anywhere in the article).

    Guidelines:

    - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
    - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
    - The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article.
    - Missing entities can appear anywhere in the new summary.
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

    Remember, use the exact same number of words for each summary.
    Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
    """

    # 日本語
    prompt = """
    記事：{要約したい文章}

    あなたは私に記事を依頼する。そして、その記事の簡潔で実体の濃い要約をどんどん作成する。

    以下の2つのステップを5回繰り返す。

    ステップ1. 先に生成された要約に欠けている、記事中の1～3個の有益なエンティティ（"; "区切り）を特定する。
    ステップ2. 同じ長さで、前の要約のすべてのエンティティと詳細をカバーし、さらに欠落しているエンティティを加えた、より密度の高い新しい要約を書く。

    欠けているエンティティとは
    - メインストーリーに関連している、
    - 具体的でありながら簡潔であること（5語以下）、
    - 新規性（前回の要約にはない）、
    - 忠実（記事中に存在する）、
    - どこにでもある（記事のどこにでもある）。

    ガイドライン

    - 最初の要約は長く（4～5文、～80語）、しかし非常に非特異的で、欠落しているとマークされたエンティティ以上の情報をほとんど含まないこと。80ワードに達するには、過度に冗長な表現とフィラー（例：「この記事は論じている」）を使用する。
    - 一語一語を大切にする：フローを改善し、エンティティを追加するスペースを作るために、前回の要約を書き直す。
    - 融合、圧縮、「この記事は論じている」のような情報量の少ないフレーズの削除でスペースを作る。
    - 要約は、高密度で簡潔でありながら自己完結的、つまり、記事なしでも容易に理解できるものにする。
    - 欠落しているエンティティは新しい要約のどこに出現してもよい。
    - 前の要約から実体を削除してはならない。スペースが確保できない場合は、新しいエンティティを少なくする。

    各要約に全く同じ語数を使用することを忘れないこと。
    JSONで答えなさい。JSONは、"Missing_Entities "と "Denser_Summary "をキーとする辞書のリスト（長さ5）でなければならない。
    """
    ```

- 参考文献
    - https://note.com/hima2b4/n/n8642c4649f32




# その他
論文などとは紐づかない(？)Prompt Engineeringに関する資料を以下にまとめています．

- [ChatGPT-Azure OpenAI 大全](https://speakerdeck.com/hirosatogamo/chatgpt-azure-openai-da-quan?slide=64)

![スクリーンショット 2023-10-25 17.00.17.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/529366/5d49e62a-5a3b-f987-2337-ec77b297c920.png)


# おわりに
今回はLLMへ入力するプロンプトに関する技術や研究の紹介でした．
最後までお読みいただきありがとうございます．記事に誤り等ありましたらご指摘いただけますと幸いです。
