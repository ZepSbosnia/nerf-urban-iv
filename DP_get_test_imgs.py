            
    for dirpath, dirnames, filenames in os.walk(ngp_dir):
        # Calculate depth from the root path
        relative_path = os.path.relpath(dirpath, ngp_dir)
        depth = relative_path.count(os.sep)
        if depth == 2:
            last_folder = os.path.basename(dirpath)
            for frame in file_paths_val:
                
                dest_val = os.path.join(data_path, 'val_'+last_folder, os.path.basename(frame))
                os.makedirs(dest_val, exist_ok=True)
                rend_img_path = os.path.join( dirpath ,'validation' , os.path.basename(frame))
                
                print('aaaa')
                print(rend_img_path)
                dgb
                os.makedirs(dest_path, exist_ok=True)
                #shutil.copy(frame, dest_path)
                #print(f"Copied {frame} -> {dest_path}") 
                val_path = dirpath + '/validation/' 
                
                print(f"Found 3-level-deep folder: {last_folder}")