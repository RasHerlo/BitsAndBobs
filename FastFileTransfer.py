
# Function to copy files into a different folder (from Bard)

# Comment on 240115: TryAndCatch needed, as many single files are corrupted
# Solution: Temporarily copy the previous functional file and log the replacement in a log-file


import os
import shutil
import time
import datetime
from ProcessLogger import log_init
import logging

def copy_files_with_substring(source_folder, destination_folder, substring):
    """
    Copies files containing a specified substring from one folder to another.

    Args:
        source_folder (str): The path to the source folder.
        destination_folder (str): The path to the destination folder.
        substring (str): The substring to search for in filenames.
    """
    print("Initiating copy_files_with_substring again again...")
    print("Current directory is: {}".format(os.getcwd()))

    folders = destination_folder.split("\\")
    logging.basicConfig(filename="".join(["logfile_", folders[-1], ".log"]), level=logging.DEBUG)

    count = 0
    errorcount = 0

    t = time.time()
    logging.info("run performed at {}".format(str(datetime.datetime.now())))
    logging.info("Source-folder = {}, Destination-folder = {}".format(source_folder, destination_folder))
    logging.info("FORMAT: corrupt 'file1' replaced by healthy 'file2': 'file1',file2'")
    for filename in os.listdir(source_folder):
        if substring in filename:
            count += 1
            try:
                source_file = os.path.join(source_folder, filename)
                destination_file = os.path.join(destination_folder, filename)
                shutil.copy2(source_file, destination_file)
                # cache the last non-corrupted file
                if "000001" in filename:
                    pl1 = filename
                if "000002" in filename:
                    pl2 = filename

                # update on progression live
                if count % 2000 == 0:
                    print("File number {} copied was {}".format(count, filename))
                    print("Time passed: {} seconds".format(time.time()-t))
            # if files are corrupt
            except:
                if "000001" in filename:
                    logging.info("{},{}".format(filename, pl1))
                    errorcount += 1
                    print("The file {} is corrupt, replaced by {}".format(filename, pl1))
                    source_file = os.path.join(source_folder, pl1)
                    shutil.copy2(source_file, destination_file)
                if "000002" in filename:
                    logging.info("{},{}".format(filename, pl2))
                    errorcount += 1
                    print("The file {} is corrupt, replaced by {}".format(filename, pl2))
                    source_file = os.path.join(source_folder, pl2)
                    shutil.copy2(source_file, destination_file)
            # if count == 400:
            #     logging.info("Breaking at file #400 after {} seconds".format(time.time()-t))
            #     break
    logging.info("Finished at file #{} after {} seconds".format(count, time.time() - t))
    logging.info("Total number of error-replacements = {}".format(errorcount))
