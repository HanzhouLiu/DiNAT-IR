import os

input_dir = "datasets/RealBlur_R/test/input"
target_dir = "datasets/RealBlur_R/test/target"

# Define the strings to remove
remove_strings = ["_blur", "_gt"]

# Rename files in the input directory
for filename in os.listdir(input_dir):
    new_name = filename
    s = remove_strings[0]
    new_name = new_name.replace(s, "")
    os.rename(os.path.join(input_dir, filename), os.path.join(input_dir, new_name))

# Rename files in the target directory
for filename in os.listdir(target_dir):
    new_name = filename
    s = remove_strings[1]
    new_name = new_name.replace(s, "")
    os.rename(os.path.join(target_dir, filename), os.path.join(target_dir, new_name))

print("Renaming completed!")
