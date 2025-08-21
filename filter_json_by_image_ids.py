import json
import os

# 画像IDリストを作成（例: GCC_train_000000000.jpg → 000000000）
image_dir = "/home/ubuntu/llava-virginia/images"
image_ids = set(
    fname.replace("GCC_train_", "").replace(".jpg", "")
    for fname in os.listdir(image_dir)
    if fname.endswith(".jpg")
)

# JSONをロード
with open("/home/ubuntu/llava-virginia/blip_laion_cc_sbu_558k.json", "r") as f:
    data = json.load(f)

# 共通IDでフィルタ
filtered = [item for item in data if item["id"] in image_ids]

# 保存
with open("/home/ubuntu/llava-virginia/blip_laion_cc_sbu_558k.filtered.json", "w") as f:
    json.dump(filtered, f, indent=2)