import os
import json
import requests
from tqdm import tqdm

json_path = "/home/ubuntu/llava-virginia/blip_laion_cc_sbu_558k.json"
save_dir = "/home/ubuntu/llava-virginia/images"
os.makedirs(save_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

for item in tqdm(data):
    url = item["image"]
    filename = os.path.join(save_dir, os.path.basename(url))
    if os.path.exists(filename):
        continue
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(filename, "wb") as out:
                out.write(r.content)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
