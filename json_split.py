import json
import numpy as np

# Parameters
directory = "/volume/datasets/complex_rand/"
used_percentage = 1  # Change this to use a different percentage of total frames

print("taking into account "+ str(used_percentage) + " percent of frames!!") 

# Load JSON
with open(directory + "transforms_all.json") as fp:
    data = json.load(fp)

# Collect all frame file paths
imgs_list = []
for d in data:
    if d == "frames":
        for frame in data[d]:
            imgs_list.append(frame["file_path"])

# Subsample a percentage of frames
imgs_list = np.array(imgs_list)
total_frames = len(imgs_list)
num_used = int(total_frames * used_percentage)

# Randomly permute and select a subset
selected_indices = np.random.permutation(total_frames)[:num_used]
imgs_list = imgs_list[selected_indices]

# Split into train, val, test
val_size = int(num_used / 10)
test_size = val_size
train_size = num_used - val_size - test_size

test_imgs = imgs_list[:test_size]
val_imgs = imgs_list[test_size:test_size + val_size]
train_imgs = imgs_list[test_size + val_size:]

# Initialize output dicts
transform_train = {}
transform_val = {}
transform_test = {}

for d in data:
    if d != "frames":
        transform_train[d] = data[d]
        transform_val[d] = data[d]
        transform_test[d] = data[d]
    else:
        transform_train["frames"] = []
        transform_val["frames"] = []
        transform_test["frames"] = []
        for frame in data[d]:
            if frame["file_path"] in test_imgs:
                transform_test["frames"].append(frame)
            elif frame["file_path"] in val_imgs:
                transform_val["frames"].append(frame)
            elif frame["file_path"] in train_imgs:
                transform_train["frames"].append(frame)

# Save the new JSONs
with open(directory + "transforms_train.json", 'w') as f:
    json.dump(transform_train, f)
with open(directory + "transforms_test.json", 'w') as f:
    json.dump(transform_test, f)
with open(directory + "transforms_val.json", 'w') as f:
    json.dump(transform_val, f)
