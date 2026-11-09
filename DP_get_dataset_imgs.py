import subprocess
import json 
import random
import os 
import shutil

global_init_path = '/home/shared/M3/Daniele/datasets'
# List of dataset paths
datasets = [ 
            '/milano_2_GPS_transp/',
            "/lego/",
            "/south_building_nosky/",
            "/Barn_nosky/",
            "/albaro_turning/",
]

ngp_dir = '/home/nas/daniele/3Dreconstruction/torch-ngp_NODIR/results/'

def ensure_png_extension(path):
    return path if path.lower().endswith('.png') or path.lower().endswith('.jpg')   else path + '.png'

# Run the command for each combination
for dataset in datasets: 
    data_path = global_init_path + dataset 
    trasnf_train = data_path + 'transforms_train.json'
    trasnf_val = data_path + 'transforms_val.json'
    trasnf_test = data_path + 'transforms_test.json'
    dest_train = data_path + 'selected_train/'
    dest_val = data_path + 'selected_val/'
    
    with open(trasnf_train, 'r') as f:
        train_json = json.load(f) 
    # Extract frames
    train_frames = train_json["frames"]

    # Randomly select 3 frames
    sampled_train_frames = random.sample(train_frames, 3)
    
    # Save new JSON
    new_json = train_json.copy()
    new_json["frames"] = sampled_train_frames

    with open(trasnf_test, "w") as f:
        json.dump(new_json, f, indent=4, ensure_ascii=False)
    
    # Extract file paths
    file_paths_train = [data_path+ frame["file_path"].lstrip("./") for frame in sampled_train_frames]
    
    with open(trasnf_val, 'r') as f:
        val_json = json.load(f) 
    # Extract frames
    val_frames = val_json["frames"]

    # Randomly select 3 frames
    sampled_val_frames = random.sample(val_frames, 3)

    # Extract file paths
    file_paths_val = [data_path+ frame["file_path"].lstrip("./") for frame in sampled_val_frames]
    
    #print(file_paths_train)
    #print(file_paths_val)
     
    if os.path.exists(dest_train):
        shutil.rmtree(dest_train)

        # Create a fresh folder
    os.makedirs(dest_train) 
    
    for frame in file_paths_train:
        #print(frame)
        dest_path = os.path.join(dest_train, os.path.basename(frame)) 
        #print(dest_path)
        
        shutil.copy(ensure_png_extension(frame), ensure_png_extension(dest_path))
        print(f"Copied {frame} -> {dest_path}") 
        
        
    if os.path.exists(dest_val):
        shutil.rmtree(dest_val)

    # Create a fresh folder
    os.makedirs(dest_val)
    for frame in file_paths_val:
        print(frame)
        dest_path = os.path.join(dest_val, os.path.basename(frame))
        #print(dest_path)
        
        shutil.copy(ensure_png_extension(frame), ensure_png_extension(dest_path))
        print(f"Copied {frame} -> {dest_path}") 
            
