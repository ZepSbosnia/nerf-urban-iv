import json
def matrix_to_tuple(matrix):
    return tuple(tuple(row) for row in matrix)

# Load the original JSON
with open('/home/shared/M3/Daniele/datasets/Barn_nosky/transforms_all.json') as f:
    data = json.load(f)

# Load the list of substrings to include from the txt file
with open('/home/shared/M3/Daniele/datasets/Barn_nosky/validation.txt') as f:
    include_list = set(line.strip() for line in f)

# Load the reference JSON (e.g., already present frames)
with open('/home/shared/M3/Daniele/datasets/Barn_nosky/transforms_test_all.json') as f:
    existing_data = json.load(f)
    existing_matrices = set(matrix_to_tuple(frame["transform_matrix"]) for frame in existing_data["frames"])


# Prepare output categories
include_txt = []
in_existing_json = []
remaining = []

for frame in data["frames"]:
    path = frame["file_path"]
    name = path.split("/")[-1].split(".")[0] 
    matrix = matrix_to_tuple(frame["transform_matrix"])
    print(name)# e.g., "r_0"
    print(include_list)
    if name in include_list:
        include_txt.append(frame)
    elif matrix in existing_matrices:
        in_existing_json.append(frame)
    else:
        remaining.append(frame)

# Save results into three separate files
base = {k: v for k, v in data.items() if k != "frames"}

with open("/home/shared/M3/Daniele/datasets/Barn_nosky/transforms_val.json", "w") as f:
    json.dump({**base, "frames": include_txt}, f, indent=4)

with open("/home/shared/M3/Daniele/datasets/Barn_nosky/transforms_test.json", "w") as f:
    json.dump({**base, "frames": in_existing_json}, f, indent=4)

with open("/home/shared/M3/Daniele/datasets/Barn_nosky/transforms_train.json", "w") as f:
    json.dump({**base, "frames": remaining}, f, indent=4)
