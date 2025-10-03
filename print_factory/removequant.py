input_path = "copy_LLaVA_notebook_no_print.py"
output_path = "copy_LLaVA_notebook_no_print_noquant.py"

in_block = False
with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
    for line in f_in:
        count = line.count('"""')
        if count == 2:
            continue  # 1行で完結するブロック
        if count == 1:
            in_block = not in_block
            continue  # ブロックの開始・終了行は書き出さない
        if in_block:
            continue  # ブロック内は書き出さない
        f_out.write(line)
print(f"Done! Output: {output_path}")