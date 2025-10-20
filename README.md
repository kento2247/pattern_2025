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
```
