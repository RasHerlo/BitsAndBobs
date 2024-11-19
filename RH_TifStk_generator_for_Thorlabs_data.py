

#!/usr/bin/env python3

# Requirements:
# Filenames should not contain any dots (except fileformat), should start with "Chan" and index number is preceded by underscore, fx. ChanA_001_001_001_001.tif
# function to identify all channel-specific tif-files in folder without Preview file
# Should work no matter if the order of magnitude of your number of frames
# Selects only the real tif files and avoids the preview files given
# Function is checking if the target folder is available, and if not,it leaves it with a message.
# This avoids any overwriting or data-mixing.

# Average of several measures indicate a speed of 11 frames/s total on a basic CPU system.
# Not set up for parallel processing yet.

# Folder selection
import tkinter as tk
from tkinter import filedialog
import os

# Image-reading
import tifffile
import numpy as np
import time
import re

def stack_tif_images(root, chan):

    # root = "C:\\Users\\svw191\\PythonFiles\\PythonTrial\\LED +APs 240926\\240926_pl100_pc001_LED+APs500microW_ex01\\"
    # chan = "ChanA"

    # Get a list of .tif files containing the search string
    tif_files = [f for f in os.listdir(root) if (f.endswith('.tif') or f.endswith('.ti')) and chan in f and 'Preview' not in f]

    first_image = tifffile.imread(os.path.join(root, tif_files[0]), key=0)  # Read the first page
    image_shape = first_image.shape

    # # # Ensure consistent image format
    for file in files:
        image = tifffile.imread(os.path.join(root, tif_files[0]), key=0)
        if image.shape != image_shape:
            print(f"Warning: Image format mismatch for {file}")
            # Implement conversion logic here (optional)

    # SOLVE PADDING ISSUE AND CHRONOLOGY FOR TIFS

    # Pad the smaller indices to match the longest one
    file_nums = []
    for file in tif_files:
        match = re.search(r"([^_\.]+)(\.[^.]+)$", file)
        file_num = match.group(1)
        file_nums.append(file_num)   

    longest_string = max(file_nums, key=len)
    max_len = len(longest_string)

    pad_files = []
    for file in tif_files:
        match = re.search(r"([^_\.]+)(\.[^.]+)$", file)
        file_num = match.group(1)
        file_num_pad = file_num.zfill(max_len)
        file = re.sub(rf"{re.escape(file_num)}\.", f"{file_num_pad}.", file)
        pad_files.append(file)

    indices = np.argsort(pad_files)

    sorted_files = []
    for i in range(len(indices)):
        sorted_files.append(tif_files[indices[i]])

    # Create an empty stack with correct data type
    image_stack = np.zeros((len(tif_files), *image_shape), dtype=first_image.dtype)

    # # # Iterate over the sorted files and add them to the stack
    for i, file in enumerate(tif_files):
        image = tifffile.imread(os.path.join(root,tif_files[indices[i]]), key=0)
        image_stack[i] = image

    print(f"Shape of stack is: {np.shape(image_stack)}")

    # Write the stacked image to a new .tif file
    tifffile.imwrite(os.path.join(root, "Data", chan, f"{chan}_stk.tif"), image_stack)
    
if __name__ == "__main__":
    # define the variables to look for:
    chans = ['ChanA','ChanB'] # make it applicable for both 1- and 2-color imaging
    
    # Select a folder
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    rtdir = filedialog.askdirectory()
    if rtdir:
        print(f"Selected folder: {rtdir}")
    
    # First Approach: Run through iterative search in each sub-branch of the folder
    # Identify the folder containing .tif files. If there are ChanA-images, check for "DATA\\ChanA"-folder. Equally with "ChanB".
    # Create folder, if they do not exist. Check if existing folders are empty, if so, generate the .tif-stack.
    
    for root, _, files in os.walk(rtdir):
        for chan in chans:
            if f"{chan}_001_001_001_001.tif" in files:
                print(f"{chan}_001_001_001_001.tif located in {root}")
                chandir = os.path.join(root, 'DATA', chan)
                if not os.path.isdir(chandir):
                    print(f"Generating {chandir}")
                    os.makedirs(chandir)                                                              
                    
                # check if stack has been made:
                if not len(os.listdir(chandir)) == 0:
                    print(f"{chandir} is not empty, skipping to avoid overwriting")
                if len(os.listdir(chandir)) == 0:
                    print(f"{chandir} is empty")
                    
                    start_time = time.time()
                    
                    stack_tif_images(root, chan)
                    print("Stack has been completed")
                    
                    end_time = time.time()
                    
                    elapsed_time = end_time - start_time
                    print("Elapsed time:", elapsed_time, "seconds")
                
