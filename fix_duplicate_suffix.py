import os
import re

# Directory containing images with duplicate suffix
directory = r"C:\Users\marco\Pictures\loras_pics\3ds_pokemon\gifs\first_frame_extracted\cropped_images\cropped_images_1024\original_dataset"

# Get all PNG files
png_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]

# Pattern to detect duplicate suffix: ends with _00001_.png_00001_.png
# We want to remove the last _00001_.png part
pattern = r'(_00001_\.png)(_00001_\.png)$'

files_to_rename = {}

for filename in png_files:
    # Check if it matches the duplicate pattern
    match = re.search(pattern, filename)
    if match:
        # Remove the last duplicate part
        new_name = re.sub(pattern, r'\1', filename)
        files_to_rename[filename] = new_name

# Print summary
print(f"Found {len(png_files)} PNG files in total")
print(f"Found {len(files_to_rename)} files with duplicate suffix to rename")
print()

if files_to_rename:
    print("Files that will be RENAMED:")
    print("-" * 80)
    for old_name, new_name in sorted(files_to_rename.items())[:20]:  # Show first 20
        print(f"{old_name}")
        print(f"  -> {new_name}")
        print()

    if len(files_to_rename) > 20:
        print(f"... and {len(files_to_rename) - 20} more files")

    print("\n" + "=" * 80)
    print(f"Total files to rename: {len(files_to_rename)}")
    print("=" * 80)

    # Execute rename
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--confirm':
        response = 'yes'
    else:
        response = 'no'
        print("\nTo rename these files, run: python fix_duplicate_suffix.py --confirm")

    if response == 'yes':
        renamed_count = 0
        for old_name, new_name in files_to_rename.items():
            old_path = os.path.join(directory, old_name)
            new_path = os.path.join(directory, new_name)

            # Check if target already exists
            if os.path.exists(new_path):
                print(f"SKIP: Target already exists: {new_name}")
                continue

            try:
                os.rename(old_path, new_path)
                renamed_count += 1
                print(f"Renamed: {old_name} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {old_name}: {e}")

        print(f"\nSuccessfully renamed {renamed_count} files")
    else:
        print("Rename cancelled")
else:
    print("No files with duplicate suffix found")
