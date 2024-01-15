
# Function to copy files into a different folder (from Bard)

import os
import shutil

def copy_folder_with_substring(source_folder, destination_folder, substring):
    """
    Copies folders containing files with the specified substring to the destination folder.
    Also counts and prints the number of identified files before copying.
    """

    print("Initiating...")

    os.makedirs(destination_folder, exist_ok=True)

    matching_file_count = 0

    for root, directories, files in os.walk(source_folder):
        for directory in directories:
            folder_path = os.path.join(root, directory)
            if any(substring in file for file in os.listdir(folder_path)):
                matching_files = [file for file in os.listdir(folder_path) if substring in file]
                matching_file_count += len(matching_files)
                print(f"Found {len(matching_files)} matching files in {folder_path}")

                source_path = os.path.join(root, directory)
                destination_path = os.path.join(destination_folder, directory)
                shutil.copytree(source_path, destination_path)

    print(f"Total of {matching_file_count} files with the substring '{substring}' were identified and copied.")
