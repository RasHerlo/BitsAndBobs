
# Transfer individual datafiles to individual folders
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog


def copy_files_with_substring_and_interval():

    # Input information from dialog-windows
    root = tk.Tk()
    root.withdraw()

    input_path = filedialog.askdirectory(title="Select a Directory with imaging files")
    input_substr = simpledialog.askstring("Input Dialog", "Enter your string:")
    input_lowerbound = simpledialog.askinteger("Input Dialog", "Enter lower-bound index:")
    input_upperbound = simpledialog.askinteger("Input Dialog", "Enter upper-bound index:")
    target_dir = filedialog.askdirectory(title="Select target Directory for folder with imaging files")
    target_fld = simpledialog.askstring("Input Dialog", "Enter name of the folder for your images:")

    if input_path:
        print("Selected input directory:", input_path)
    if input_substr:
        print("Substring entered:", input_substr)

    root.destroy()  # Close the hidden window

    # Identify the files and confirm the transfer




