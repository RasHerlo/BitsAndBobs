"""
Script to organize Lemon and Lychee imaging files from time-stamped folders.

This script:
1. Scans time-stamp folders (t_0h, t_2h, etc.) in the root directory
2. Moves non-BF files to Lemon/ or Lychee/ folders based on filename prefix
3. Appends the timestamp to filenames (e.g., _00h, _02h, _24h)
"""

import os
import shutil
import re
from pathlib import Path


def extract_timestamp(folder_name: str) -> str:
    """
    Extract timestamp from folder name and format as 2-digit hours.
    
    Examples:
        t_0h  -> 00h
        t_2h  -> 02h
        t_24h -> 24h
    """
    match = re.match(r't_(\d+)h', folder_name)
    if match:
        hours = int(match.group(1))
        return f"{hours:02d}h"
    return None


def should_exclude(filename: str) -> bool:
    """Check if file should be excluded (contains BF anywhere in name)."""
    return "BF" in filename


def get_destination_folder(filename: str) -> str:
    """Determine destination folder based on filename prefix."""
    if filename.startswith("Lemon"):
        return "Lemon"
    elif filename.startswith("Lychee"):
        return "Lychee"
    return None


def rename_with_timestamp(filename: str, timestamp: str) -> str:
    """
    Insert timestamp suffix before the file extension.
    
    Example:
        Lemon_LL_200ms_pre_Im1.nd2 + 00h -> Lemon_LL_200ms_pre_Im1_00h.nd2
    """
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"


def organize_files(root_dir: str, dry_run: bool = False):
    """
    Main function to organize files from time-stamp folders.
    
    Args:
        root_dir: Path to the root directory (F:\Klelia)
        dry_run: If True, only print what would be done without moving files
    """
    root_path = Path(root_dir)
    
    # Create destination folders
    lemon_dir = root_path / "Lemon"
    lychee_dir = root_path / "Lychee"
    
    if not dry_run:
        lemon_dir.mkdir(exist_ok=True)
        lychee_dir.mkdir(exist_ok=True)
        print(f"Created directories: {lemon_dir}, {lychee_dir}")
    
    # Statistics
    stats = {
        "moved": 0,
        "skipped_bf": 0,
        "skipped_unknown": 0,
        "errors": 0
    }
    
    # Find all time-stamp folders
    timestamp_folders = sorted([
        d for d in root_path.iterdir() 
        if d.is_dir() and re.match(r't_\d+h', d.name)
    ])
    
    print(f"\nFound {len(timestamp_folders)} time-stamp folders: {[f.name for f in timestamp_folders]}")
    
    for folder in timestamp_folders:
        timestamp = extract_timestamp(folder.name)
        if not timestamp:
            print(f"  Warning: Could not extract timestamp from {folder.name}")
            continue
            
        print(f"\nProcessing {folder.name} (timestamp: {timestamp})...")
        
        # Process each file in the folder
        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue
                
            filename = file_path.name
            
            # Skip BF files
            if should_exclude(filename):
                stats["skipped_bf"] += 1
                if dry_run:
                    print(f"  [SKIP-BF] {filename}")
                continue
            
            # Determine destination
            dest_folder_name = get_destination_folder(filename)
            if not dest_folder_name:
                stats["skipped_unknown"] += 1
                print(f"  [SKIP-UNKNOWN] {filename} (not Lemon or Lychee)")
                continue
            
            # Build new filename and destination path
            new_filename = rename_with_timestamp(filename, timestamp)
            dest_folder = lemon_dir if dest_folder_name == "Lemon" else lychee_dir
            dest_path = dest_folder / new_filename
            
            if dry_run:
                print(f"  [MOVE] {filename} -> {dest_folder_name}/{new_filename}")
            else:
                try:
                    shutil.move(str(file_path), str(dest_path))
                    stats["moved"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    print(f"  [ERROR] Failed to move {filename}: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    if dry_run:
        print("(DRY RUN - no files were actually moved)")
    print(f"  Files moved:        {stats['moved']}")
    print(f"  Skipped (BF):       {stats['skipped_bf']}")
    print(f"  Skipped (unknown):  {stats['skipped_unknown']}")
    print(f"  Errors:             {stats['errors']}")
    

if __name__ == "__main__":
    ROOT_DIR = r"F:\Klelia"
    
    # First, do a dry run to preview what will happen
    print("="*50)
    print("DRY RUN - Preview of changes")
    print("="*50)
    organize_files(ROOT_DIR, dry_run=True)
    
    # Ask for confirmation
    print("\n")
    response = input("Proceed with moving files? (yes/no): ")
    
    if response.lower() in ["yes", "y"]:
        print("\n" + "="*50)
        print("EXECUTING FILE MOVES")
        print("="*50)
        organize_files(ROOT_DIR, dry_run=False)
        print("\nDone!")
    else:
        print("Operation cancelled.")
