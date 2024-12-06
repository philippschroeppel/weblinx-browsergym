"""
Given WebLINX-full, this will zip everything in demonstrations and move them to a new directory.
"""
import os
import json
import zipfile

from tqdm import tqdm

# Define the base directory
base_dir = os.path.expanduser('~/scratch/WebLINX-full')
source_dir = os.path.join(base_dir, 'demonstrations')
output_dir = os.path.join(base_dir, 'demonstrations_zip')

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create a done.json file to indicate that the zipping is complete for a given directory, so we don't zip it again
done_file = os.path.join('./weblinx_zipping_done.json')
if not os.path.exists(done_file):
    done = {}
else:
    with open(done_file, 'r') as f:
        done = json.load(f)

# Iterate over all the items in the source directory
pbar = tqdm(os.listdir(source_dir))
for folder_name in pbar:
    # update tqdm to show the folder name
    pbar.set_description(f'Zipping: {folder_name}')

    folder_path = os.path.join(source_dir, folder_name)

    # Only zip directories
    if os.path.isdir(folder_path):
        zip_filename = os.path.join(output_dir, f"{folder_name}.zip")

        # if the zip file already exists, skip this folder
        if folder_name in done:
            continue
        
        # Create the zip file
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through the directory and add all files to the zip
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to the zip, but keep the folder structure
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
        
        # now that it's done, we add the folder to the done list
        done[folder_name] = zip_filename
        with open(done_file, 'w') as f:
            json.dump(done, f)
        


print("Zipping complete!")
print(f"Zipped files are stored in {output_dir}")