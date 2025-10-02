input_path = "LLaVA_notebook_cleansed.py"
output_path = "LLaVA_notebook_no_print.py"

with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
    for line in f_in:
        if "print(" in line and "def" not in line:
            continue
        f_out.write(line)
print(f"Done! Output: {output_path}")