import subprocess

# List of dataset paths
datasets = [ 
            '/datasets/milano_2_GPS_transp',
            "/datasets/lego",
            "/datasets/south_building",
            "/datasets/Barn_nosky",
            "/datasets/albaro_turning",
]


# Optimizer options: opt_name -> list of values
opt_options = {
    None: [None],           # No optimizer specified
    "MS_lambda": [1e-3, 5e-4],
    "TV_lambda": [1e-2, 5e-3],
    "l1_loss" : [1e-2, 5e-3],
}

# Run the command for each combination
for dataset in datasets:
    for opt_name, opt_values in opt_options.items():
        for opt_value in opt_values:
            # Construct workspace name
            dataset_name = dataset.rstrip('/').split('/')[-1]
            workspace_name = f"results/{dataset_name}"

            # Construct the command
            cmd = [
                "python3", "main_nerf.py", dataset,
                "-O",
                "--workspace", workspace_name, 
                "--test",  
            ]

            # Add optimizer if defined
            if opt_name:
                cmd += ["--"+str(opt_name), str(opt_value)]

            # Print and run
            print("Running:", " ".join(cmd))
            subprocess.run(cmd)
