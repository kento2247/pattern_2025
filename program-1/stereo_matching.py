# -*- coding: utf-8 -*-

# ステレオマッチング

import numpy as np
from PIL import Image
from tqdm import tqdm

# パラメータ設定
left_file = "left1.jpg"
right_file = "right1.jpg"
search_size = 21 // 2  # 探索領域の大きさ
template_size = 21 // 2  # テンプレートの大きさ

# 画像の読み込み
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
            for dy in range(-search_size, search_size + 1):
                # for dy in [0]:  # 水平方向のみに制限
                r_x = x + dx
                r_y = y + dy

                r_x1, r_y1 = max(0, r_x - template_size), max(0, r_y - template_size)
                r_x2, r_y2 = min(right_width - 1, r_x + template_size), min(
                    right_height - 1, r_y + template_size
                )

                if (r_x2 - r_x1) != (l_x2 - l_x1) or (r_y2 - r_y1) != (l_y2 - l_y1):
                    continue

                region = right[r_y1 : r_y2 + 1, r_x1 : r_x2 + 1]

                diff = template - region
                ssd = np.sum(diff * diff)
                # sad = np.sum(np.abs(diff))

                if ssd < min_val:
                    min_val = ssd
                    ans_x = r_x
                    ans_y = r_y

        result[y, x] = (x - ans_x) * (x - ans_x) + (y - ans_y) * (y - ans_y)

# 視差マップのグレースケール化
min_value = np.min(result)
max_value = np.max(result)
result = (result - min_value) / (max_value - min_value) * 255

# 視差マップの保存
result_img = Image.fromarray(np.uint8(result))
result_img.save("result.jpg")
