import argparse
import os

import numpy as np
from PIL import Image


def load_images(data_dir, num_images=100):
    """MNISTデータをディレクトリ構造から読み込む"""
    classes = {}
    for digit in range(10):
        digit_dir = os.path.join(data_dir, str(digit))
        images = []
        for filename in sorted(os.listdir(digit_dir))[:num_images]:
            if filename.endswith('.jpg'):
                img_path = os.path.join(digit_dir, filename)
                img = Image.open(img_path).convert('L')  # グレースケールに変換
                img_array = np.array(img).flatten()  # 1次元配列に変換
                images.append(img_array)
        classes[digit] = np.array(images)
    return classes


def create_prototypes(train_data, K, N):
    """各クラスについてK個のプロトタイプを作成（N個の画像の平均）"""
    prototypes = {}
    for digit, images in train_data.items():
        prototypes[digit] = []
        for k in range(K):
            # N個の画像をランダムに選択
            indices = np.random.choice(len(images), N, replace=False)
            selected_images = images[indices]
            # 平均画像をプロトタイプとする
            prototype = np.mean(selected_images, axis=0)
            prototypes[digit].append(prototype)
    return prototypes


def euclidean_distance(x, y):
    """ユークリッド距離を計算"""
    return np.sqrt(np.sum((x - y) ** 2))


def classify(test_image, prototypes):
    """最近傍法により画像を分類"""
    min_distance = float('inf')
    predicted_class = -1

    # 全てのプロトタイプとの距離を計算
    for digit, prototype_list in prototypes.items():
        for prototype in prototype_list:
            distance = euclidean_distance(test_image, prototype)
            if distance < min_distance:
                min_distance = distance
                predicted_class = digit

    return predicted_class


def print_confusion_matrix(confusion_matrix):
    """混同行列を表示"""
    print("\n [混同行列]")
    print(confusion_matrix)
    print("\n 正解数 ->", np.trace(confusion_matrix))


def main(args):
    """メイン関数"""
    np.random.seed(42)

    train_dir = os.path.join(args.program_dir, "mnist", "train")
    test_dir = os.path.join(args.program_dir, "mnist", "test")

    print(f"学習データを読み込み中... ({train_dir})")
    train_data = load_images(train_dir, args.image_num)

    print(f"各クラスについて{args.K}個のプロトタイプを作成中（各プロトタイプは{args.N}枚の画像の平均）...")
    prototypes = create_prototypes(train_data, args.K, args.N)

    print(f"テストデータを読み込み中... ({test_dir})")
    test_data = load_images(test_dir, num_images=1000)

    print("テスト画像を分類中...")
    # 混同行列の初期化（10x10）
    confusion_matrix = np.zeros((10, 10), dtype=int)
    correct = 0
    total = 0

    for true_digit, test_images in test_data.items():
        for test_image in test_images:
            predicted_digit = classify(test_image, prototypes)
            # 混同行列を更新
            confusion_matrix[true_digit][predicted_digit] += 1
            if predicted_digit == true_digit:
                correct += 1
            total += 1

    # 結果の表示
    accuracy = (correct / total) * 100
    print(f"\n========== 認識結果 ==========")
    print(f"パラメータ: K={args.K}, N={args.N}")
    print(f"テスト画像数: {total}")
    print(f"正解数: {correct}")
    print(f"認識率: {accuracy:.2f}%")

    # 混同行列の表示
    print_confusion_matrix(confusion_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_dir", default="program-3")
    parser.add_argument("--train_dir", default="data/MNIST/train")
    parser.add_argument("--test_dir", default="data/MNIST/test")
    parser.add_argument("--image_num", type=int, default=100, help="Number of training images per class")
    parser.add_argument("--K", type=int, default=3, help="Number of prototypes per class")
    parser.add_argument("--N", type=int, default=5, help="Number of images per prototype")
    args = parser.parse_args()
    main(args)