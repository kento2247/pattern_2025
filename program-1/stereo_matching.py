# -*- coding: utf-8 -*-

# ステレオマッチング

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

# パラメータ設定
parser = argparse.ArgumentParser()
parser.add_argument("--program_dir", default="program-1")
parser.add_argument("--result_dir", default="program-1/results")
parser.add_argument("--left", default="left1.jpg")
parser.add_argument("--right", default="right1.jpg")
parser.add_argument("--sim_metric", default="ssd")
parser.add_argument("--explore_y", action="store_true")
args = parser.parse_args()

program_dir = args.program_dir
left_image_name = args.left
right_image_name = args.right
similarity_metric = args.sim_metric  # "ssd" または "sad"
explore_y = False  # y方向にも探索する場合はTrueに設定
search_size = 21 // 2  # 探索領域の大きさ
template_size = 21 // 2  # テンプレートの大きさ

# 画像の読み込み
left_file = os.path.join(program_dir, left_image_name)
right_file = os.path.join(program_dir, right_image_name)
left_img = Image.open(left_file).convert("L")
right_img = Image.open(right_file).convert("L")
left = np.asarray(left_img).astype(np.float32)
right = np.asarray(right_img).astype(np.float32)
left_height, left_width = left.shape
right_height, right_width = right.shape
print("left.shape: ", left.shape)
print("right.shape: ", right.shape)

result = np.ones((left_height, left_width))  # 視差マップの初期化

for y in tqdm(range(0, left_height, 1)):
    for x in range(0, left_width, 1):
        ans_x, ans_y = 0, 0
        min_val = float("inf")

        l_x1, l_y1 = max(0, x - template_size), max(0, y - template_size)
        l_x2, l_y2 = min(left_width - 1, x + template_size), min(
            left_height - 1, y + template_size
        )
        template = left[l_y1 : l_y2 + 1, l_x1 : l_x2 + 1]

        for dx in range(-search_size, search_size + 1):
            y_search_range = range(-search_size, search_size + 1) if explore_y else [0]
            for dy in y_search_range:
                r_x = x + dx
                r_y = y + dy

                r_x1, r_y1 = max(0, r_x - template_size), max(0, r_y - template_size)
                r_x2, r_y2 = min(right_width - 1, r_x + template_size), min(
                    right_height - 1, r_y + template_size
                )

                if (r_x2 - r_x1) != (l_x2 - l_x1) or (r_y2 - r_y1) != (l_y2 - l_y1):
                    continue

                region = right[r_y1 : r_y2 + 1, r_x1 : r_x2 + 1]

                flatten_template = template.flatten()
                flatten_region = region.flatten()
                if similarity_metric == "ssd":
                    score = np.dot(
                        (flatten_template - flatten_region).T,
                        (flatten_template - flatten_region),
                    )
                elif similarity_metric == "sad":
                    score = np.sum(np.abs(flatten_template - flatten_region))

                if score < min_val:
                    min_val = score
                    ans_x = r_x
                    ans_y = r_y

        result[y, x] = (x - ans_x) * (x - ans_x) + (y - ans_y) * (y - ans_y)

# 視差マップのグレースケール化
min_value = np.min(result)
max_value = np.max(result)
result = (result - min_value) / (max_value - min_value) * 255

# 視差マップの保存
result_img = Image.fromarray(np.uint8(result))
save_file_name = f"{similarity_metric}_{'exploreY' if explore_y else 'noExploreY'}.jpg"
save_path = os.path.join(args.result_dir, save_file_name)
result_img.save(save_path)
