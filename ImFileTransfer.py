
# Transfer individual datafiles to individual folders
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog


def copy_files_with_substring_and_interval():

    # enter all info with input-dialogs
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import simpledialog
    from tkinter import messagebox
    import os
    import shutil

    root = tk.Tk()
    root.withdraw()

    input_path = filedialog.askdirectory(title="Select a Directory with imaging files")
    input_substr = simpledialog.askstring("Input Dialog", "Enter your substring:")
    lowerbound = simpledialog.askinteger("Input Dialog", "Enter lower-bound index:")
    upperbound = simpledialog.askinteger("Input Dialog", "Enter upper-bound index:")
    target_dir = filedialog.askdirectory(title="Select target Directory for folder with imaging files")
    target_fld = simpledialog.askstring("Input Dialog", "Enter name of the folder for your images:")

    if input_path:
        print("Selected input directory:", input_path)
    if input_substr:
        print("Substring entered:", input_substr)

    root.destroy()  # Close the hidden window

    # Identify the files and confirm the transfer
    file_paths = [
        filename
        for filename in os.listdir(input_path)
        if input_substr in filename
    ]
    file_paths = file_paths[lowerbound:upperbound]
    msg_process = f"""The first file transferred will be: {file_paths[0]}, and the last will be {file_paths[-1]}.
                  Do you want to proceed?"""

    root = tk.Tk()
    root.withdraw()

    answer = messagebox.askyesno("Confirmation", msg_process)

    root.destroy()

    # if answer is yes, start the transfer
    if answer:
        target_new = os.path.join(target_dir, target_fld)
        os.mkdir(target_new)

        for imfile in file_paths:
            source_file = os.path.join(input_path, imfile)
            destination_file = os.path.join(target_new, imfile)
            shutil.copy2(source_file, destination_file)




