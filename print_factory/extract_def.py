input_path = "notebook_forward_CC3M_2.log"
output_path = "notebook_forward_CC3M_2_def.log"

with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
    for line in f_in:
        if line.lstrip().startswith("def "):
            f_out.write(line)
print(f"Done! Output: {output_path}")