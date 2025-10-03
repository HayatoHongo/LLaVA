input_path = "copy_LLaVA_notebook_no_print_noquant.py"
output_path = "copy_LLaVA_notebook_no_print_noquant_nospace.py"

prev_blank = False
with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
    for line in f_in:
        if line.strip() == "":
            if prev_blank:
                continue  # 連続する空行はスキップ
            prev_blank = True
        else:
            prev_blank = False
        f_out.write(line)
print(f"Done! Output: {output_path}")