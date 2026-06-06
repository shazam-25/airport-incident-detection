from pathlib import Path

def print_directory_tree(root_path, prefix=""):
    """
    TASK:
    Helper function to print the directory structure.
    """
    path = Path(root_path)
    # Filter to only include directories
    folders = [f for f in path.iterdir() if f.is_dir()]
    folders.sort(key=lambda x: x.name.lower())

    for i, folder in enumerate(folders):
        is_last = (i == len(folders) - 1)
        connector = "└── " if is_last else "├── "

        print(f"{prefix}{connector}{folder.name}")

        # Recurse into the subfolder
        new_prefix = prefix + ("    " if is_last else "│   ")
        print_directory_tree(folder, new_prefix)