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

## 実行方法

```sh
uv run ファイル名
```