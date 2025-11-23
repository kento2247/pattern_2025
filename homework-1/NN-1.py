# 最近傍法（プロトタイプが複数個）

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def main(args):
    # プロトタイプを格納する配列
    train_num = args.train_num
    train_img = np.zeros((10, train_num, 28, 28), dtype=np.float32)

    # プロトタイプの読み込み
    for num_index in tqdm(range(10), desc="Loading training images"):
        for j in range(1, train_num + 1):
            train_file = os.path.join(
                args.image_dir, "train", str(num_index), f"{num_index}_{j}.jpg"
            )
            train_img[num_index][j - 1] = np.asarray(
                Image.open(train_file).convert("L")
            ).astype(np.float32)

    # 混同行列
    result = np.zeros((10, 10), dtype=np.int32)
    for num_index in tqdm(range(10), desc="Classifying test images"):
        for j in range(1, 101):
            # 未知パターンの読み込み
            pat_file = os.path.join(
                args.image_dir, "test", str(num_index), f"{num_index}_{j}.jpg"
            )
            pat_img = np.asarray(Image.open(pat_file).convert("L")).astype(np.float32)

            dist_list = []

            # 最近傍法
            for k in range(10):
                for l in range(1, train_num + 1):
                    # SSDの計算
                    t = train_img[k][l - 1].flatten()
                    p = pat_img.flatten()
                    dist = np.dot((t - p).T, (t - p))

                    dist_list.append((dist, k))

            # 距離の小さい順にソート
            dist_list.sort(key=lambda x: x[0])
            # 上位k個のクラスを取得
            top_k = dist_list[: args.k]

            # 結果の出力
            for class_id in top_k:
                result[num_index][class_id[1]] += 1

    print("\n [ 混同行列 ]")
    print(result)
    print("\n 正解数 -> ", np.trace(result))

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program_dir", type=str, default="program-2", help="Program directory"
    )
    parser.add_argument(
        "--image_dir", type=str, default="program-2/mnist", help="Image directory"
    )
    parser.add_argument(
        "--train_num", type=int, default=100, help="Number of training images per class"
    )
    parser.add_argument("--k", type=int, default=1, help="Number of nearest neighbors")
    args = parser.parse_args()
    main(args)