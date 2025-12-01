# -*- coding: utf-8 -*-

"""
◼ adaline.pyをオリジナル（?）パーセプトロン（スライド52ペー
ジ）に改良しなさい．
◼ adaline.pyは，A層，R層を対象とした二層のネットワークで
す．
◼ S層とA層の結合を追加し，三層のネットワークとして下さい．
❑ 結合係数は，ランダムに-1，1と固定して下さい
❑ この二つの層間の結合係数は学習しなくてよい．
◼ 活性化関数にステップ関数を導入して下さい．
◼ 学習は収束しませんので，適当な回数で停止して下さい．
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ステップ関数
def step_function(x):
    return np.where(x >= 0, 1, 0)


# A層（連合層）- S層からの入力を受け取る
class Aunit:
    def __init__(self, input_size, hidden_size):
        # S層→A層の結合係数（-1または1でランダム固定、学習しない）
        self.w = np.random.choice([-1, 1], size=(input_size, hidden_size))
        # 閾値
        self.b = np.zeros(hidden_size)

    def Propagation(self, x):
        # 内部状態
        u = np.dot(x, self.w) + self.b
        # ステップ関数で活性化
        self.out = step_function(u)
        return self.out


# R層（出力層）
class Outunit:
    def __init__(self, m, n):
        # 重み
        self.w = np.random.uniform(-0.5, 0.5, (m, n))
        # 閾値
        self.b = np.random.uniform(-0.5, 0.5, n)

    def Propagation(self, x):
        self.x = x
        # 内部状態
        self.u = np.dot(self.x, self.w) + self.b
        # ステップ関数で活性化
        self.out = step_function(self.u)

    def Error(self, t):
        # 誤差
        delta = self.out - t
        # 重み，閾値の修正値
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)

    def Update_weight(self, alpha):
        # 重み，閾値の修正
        self.w -= alpha * self.grad_w
        self.b -= alpha * self.grad_b

    def Save(self, save_dir, class_num, hidden_size):
        save_path = os.path.join(save_dir, "perceptron.npz")
        os.makedirs(save_dir, exist_ok=True)

        # 重み，閾値の保存
        np.savez(save_path, w=self.w, b=self.b)

        # 重みの画像化
        for i in range(class_num):
            side = int(np.sqrt(hidden_size))
            if side * side == hidden_size:
                a = np.reshape(self.w[:, i], (side, side))
                plt.imshow(a, interpolation="nearest")
                file = os.path.join(save_dir, f"weight-{i}.png")
                plt.savefig(file)
                plt.close()

    def Load(self, filename):
        # 重み，閾値のロード
        work = np.load(filename)
        self.w = work["w"]
        self.b = work["b"]


# データの読み込み
def Read_data(mnist_dir, split, class_num, train_num, size, feature):

    data_vec = np.zeros((class_num, train_num, feature), dtype=np.float64)

    for i in range(class_num):
        for j in range(1, train_num + 1):
            # グレースケール画像で読み込み→大きさの変更→numpyに変換，ベクトル化
            train_file = os.path.join(
                mnist_dir,
                split,
                str(i),
                f"{i}_{j}.jpg",
            )
            work_img = Image.open(train_file).convert("L")
            resize_img = work_img.resize((size, size))
            data_vec[i][j - 1] = np.asarray(resize_img).astype(np.float64).flatten()

            # 入力値の合計を1とする
            data_vec[i][j - 1] = data_vec[i][j - 1] / np.sum(data_vec[i][j - 1])

    return data_vec


# 予測
def Predict(aunit, outunit, data_vec, class_num, train_num, feature):
    # 混同行列
    result = np.zeros((class_num, class_num), dtype=np.int32)

    for i in range(class_num):
        for j in range(0, train_num):
            # 入力データ（S層）
            input_data = data_vec[i][j].reshape(1, feature)

            # S層→A層の伝播
            a_out = aunit.Propagation(input_data)

            # A層→R層の伝播
            outunit.Propagation(a_out)

            # 予測
            ans = np.argmax(outunit.out[0])

            result[i][ans] += 1
            print(i, j, "->", ans)

    print("\n [混同行列]")
    print(result)
    print("\n 正解数 ->", np.trace(result))


# 学習
def Train(
    aunit,
    outunit,
    data_vec,
    class_num,
    train_num,
    feature,
    hidden_size,
    alpha,
    epoch,
    save_dir,
):

    for e in range(epoch):
        error = 0.0
        for i in range(class_num):
            for j in range(0, train_num):
                # 入力データ（S層）
                rnd_c = np.random.randint(class_num)
                rnd_n = np.random.randint(train_num)
                input_data = data_vec[rnd_c][rnd_n].reshape(1, feature)

                # S層→A層の伝播
                a_out = aunit.Propagation(input_data)

                # A層→R層の伝播
                outunit.Propagation(a_out)

                # 教師信号
                teach = np.zeros((1, class_num))
                teach[0][rnd_c] = 1

                # 誤差
                outunit.Error(teach)

                # 重みの修正（A層→R層のみ）
                outunit.Update_weight(alpha)

                # 誤差二乗和
                error += np.sum((outunit.out - teach) ** 2)
        print(e, "->", error)

    # 重みの保存
    outunit.Save(save_dir, class_num, hidden_size)


def main():
    parser = argparse.ArgumentParser(description="Original Perceptron for MNIST")
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument(
        "--mnist_dir", type=str, default="data/mnist", help="MNIST directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/program-7",
        help="保存先パス",
    )
    parser.add_argument("--class_num", type=int, default=10, help="クラス数")
    parser.add_argument("--size", type=int, default=14, help="画像の大きさ")
    parser.add_argument("--train_num", type=int, default=100, help="学習データ数")
    parser.add_argument("--hidden_size", type=int, default=100, help="A層のユニット数")
    parser.add_argument("--alpha", type=float, default=0.1, help="学習係数")
    parser.add_argument("--epoch", type=int, default=100, help="エポック数")
    args = parser.parse_args()

    # パラメータの設定
    class_num = args.class_num
    size = args.size
    feature = size * size
    train_num = args.train_num
    hidden_size = args.hidden_size
    alpha = args.alpha
    epoch = args.epoch

    split = "train" if args.mode == "train" else "test"
    data_vec = Read_data(args.mnist_dir, split, class_num, train_num, size, feature)

    # A層のコンストラクター（S層→A層、結合係数は固定）
    aunit = Aunit(feature, hidden_size)

    # R層（出力層）のコンストラクター
    outunit = Outunit(hidden_size, class_num)

    # 引数がtrainの場合
    if args.mode == "train":
        # 学習
        Train(
            aunit,
            outunit,
            data_vec,
            class_num,
            train_num,
            feature,
            hidden_size,
            alpha,
            epoch,
            args.save_dir,
        )

    # 引数がpredictの場合
    elif args.mode == "predict":
        # テストデータの予測
        outunit.Load(os.path.join(args.save_dir, "perceptron.npz"))
        Predict(aunit, outunit, data_vec, class_num, train_num, feature)


if __name__ == "__main__":
    main()
