import os
import re

# Directory to clean
directory = r"C:\Users\marco\Pictures\loras_pics\3ds_pokemon\gifs\first_frame_extracted\original_dataset"

# Get all PNG files
png_files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]

# Pattern to detect base names (without suffixes like -2, -3, -f, -mega, etc)
# Example: "pikachu-f.png" -> base would be "pikachu.png"
# Example: "abra2.png" -> base would be "abra.png"

files_to_delete = []
base_files = {}

for filename in png_files:
    name_without_ext = os.path.splitext(filename)[0]

    # Try to find the base name by removing common suffixes
    # Pattern: ends with digit(s) or -something
    base_name = re.sub(r'(-\w+|\d+)$', '', name_without_ext)

    # If the base name is different from current name, this might be a variation
    if base_name != name_without_ext:
        base_file = base_name + '.png'

        # EXCEPTION: Skip files with "mega" in the suffix (keep mega evolutions)
        if 'mega' in name_without_ext.lower() and 'mega' not in base_name.lower():
            continue

        # Check if the base file exists
        if base_file in png_files:
            # This is a variation and base exists - mark for deletion
            files_to_delete.append(filename)
            if base_file not in base_files:
                base_files[base_file] = []
            base_files[base_file].append(filename)

# Print summary
print(f"Found {len(png_files)} PNG files in total")
print(f"Found {len(files_to_delete)} variation files to delete")
print()

if files_to_delete:
    print("Files that will be DELETED (variations):")
    print("-" * 60)
    for base_file in sorted(base_files.keys()):
        print(f"\nBase: {base_file} (KEEP)")
        for variant in sorted(base_files[base_file]):
            print(f"  -> {variant} (DELETE)")

    print("\n" + "=" * 60)
    print(f"Total files to delete: {len(files_to_delete)}")
    print("=" * 60)

    # Ask for confirmation
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--confirm':
        response = 'yes'
    else:
        response = 'no'
        print("\nTo delete these files, run: python cleanup_variations.py --confirm")

    if response == 'yes':
        deleted_count = 0
        for filename in files_to_delete:
            filepath = os.path.join(directory, filename)
            try:
                os.remove(filepath)
                deleted_count += 1
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")

        print(f"\nSuccessfully deleted {deleted_count} files")
    else:
        print("Deletion cancelled")
else:
    print("No variation files found to delete")
