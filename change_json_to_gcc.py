import json

with open("/home/ubuntu/llava-virginia/blip_laion_cc_sbu_558k_filtered.json") as f:
    data = json.load(f)

for item in data:
    item["image"] = f"GCC_train_{item['id']}.jpg"

with open("/home/ubuntu/llava-virginia/blip_laion_cc_sbu_558k_filtered.json", "w") as f:
    json.dump(data, f, indent=2)