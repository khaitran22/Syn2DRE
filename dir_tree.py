import os

def print_tree(startpath, prefix=""):
    for idx, item in enumerate(sorted(os.listdir(startpath))):
        path = os.path.join(startpath, item)
        connector = "└── " if idx == len(os.listdir(startpath)) - 1 else "├── "
        print(prefix + connector + item)
        if os.path.isdir(path):
            extension = "    " if idx == len(os.listdir(startpath)) - 1 else "│   "
            print_tree(path, prefix + extension)

# Example usage:
root_dir = "/scratch/user/uqptran9/Code/Work_3/dev/dataset/"  # change this to your actual root path
print("root")
print_tree(root_dir)