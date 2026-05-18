import os
import re
import shutil
import glob
import csv

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AC_DIR = os.path.join(BASE_DIR, 'ac')
DATA_DIR = os.path.join(BASE_DIR, 'data')
METADATA_PATH = os.path.join(DATA_DIR, 'metadata.csv')

def main():
    if not os.path.exists(AC_DIR):
        print(f"Error: Source directory '{AC_DIR}' does not exist.")
        return
    if not os.path.exists(DATA_DIR):
        print(f"Error: Destination directory '{DATA_DIR}' does not exist.")
        return
    if not os.path.exists(METADATA_PATH):
        print(f"Error: Metadata file '{METADATA_PATH}' does not exist.")
        return

    # 1. Find the maximum existing index in data/
    existing_files = glob.glob(os.path.join(DATA_DIR, 'map_*.png'))
    max_idx = -1
    pattern = re.compile(r'map_(\d{5})\.png')
    
    for f in existing_files:
        basename = os.path.basename(f)
        match = pattern.match(basename)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx

    next_idx = max_idx + 1
    print(f"Current max index in data: {max_idx:05d}")
    print(f"Starting import from index: {next_idx:05d}")

    # 2. Scan map_*.png in ac/
    ac_files = glob.glob(os.path.join(AC_DIR, 'map_*.png'))
    if not ac_files:
        print("No files matching 'map_*.png' found in 'ac/' directory.")
        return

    ac_files.sort()
    print(f"Found {len(ac_files)} files in 'ac/' to import.")

    imported_count = 0
    new_rows = []

    # Check if metadata.csv ends with a newline
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
        has_newline = content.endswith('\n') or content.endswith('\r')

    label_pattern = re.compile(r'map_([A-Z0-9]+)\.png', re.IGNORECASE)

    for src_path in ac_files:
        basename = os.path.basename(src_path)
        match = label_pattern.match(basename)
        if not match:
            print(f"Skipping {basename}: filename does not match pattern map_[LABEL].png")
            continue
        
        label = match.group(1)
        new_filename = f"map_{next_idx:05d}.png"
        dst_path = os.path.join(DATA_DIR, new_filename)

        try:
            shutil.copy2(src_path, dst_path)
            new_rows.append((new_filename, label))
            next_idx += 1
            imported_count += 1
        except Exception as e:
            print(f"Error copying file {basename}: {e}")
            break

    # 3. Update metadata.csv
    if imported_count > 0:
        with open(METADATA_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not has_newline:
                f.write('\n')
            for row in new_rows:
                writer.writerow(row)
        
        print(f"--> Success: Imported {imported_count} images to '{DATA_DIR}' and updated '{METADATA_PATH}'.")
        print("You can safely delete files in 'ac/' directory now.")
    else:
        print("No images were imported.")

if __name__ == '__main__':
    main()
