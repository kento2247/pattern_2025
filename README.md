# pattern_2025

## 環境構築

### uv をインストール

uv をインストールするには、以下のコマンドを実行してください。windows 用の手順については、[公式ドキュメント](https://docs.astral.sh/uv/getting-started/installation/)をご覧ください。

```sh
# for MacOS or Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Python/ライブラリインストール

```sh
uv sync
```

## 課題 1

ステレオマッチングの実装

### 実行コマンド例

```sh
# 類似度スコアの計算に SSD を使用．Y軸探索なし
uv run python program-1/stereo_matching.py --program_dir program-1 --result_dir program-1/results --left left1.jpg --right right1.jpg --sim_metric ssd

# 類似度スコアの計算に SSD を使用．Y軸探索あり
uv run python program-1/stereo_matching.py --program_dir program-1 --result_dir program-1/results --left left1.jpg --right right1.jpg --sim_metric ssd --explore_y

# 類似度スコアの計算に SAD を使用．Y軸探索なし
uv run python program-1/stereo_matching.py --program_dir program-1 --result_dir program-1/results --left left1.jpg --right right1.jpg --sim_metric sad

# 類似度スコアの計算に SAD を使用．Y軸探索あり
uv run python program-1/stereo_matching.py --program_dir program-1 --result_dir program-1/results --left left1.jpg --right right1.jpg --sim_metric sad --explore_y
```

## 課題 2

### データセットのダウンロード

以下のコマンドを実行して、データセットをダウンロードしてください。

```sh
wget -P program-2 http://lecture.comp.ae.keio.ac.jp/pattern2025/pdf/mnist.zip
unzip program-2/mnist.zip -d program-2/mnist
rm program-2/mnist.zip
```

### 実行コマンド例

```sh
uv run python program-2/NN-1.py --program_dir program-2 --image_dir program-2/mnist --train_num 100 --k 3
```
