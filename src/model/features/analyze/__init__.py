import os

def current_dir_path_for(path):
    current_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(current_dir_path):
        os.makedirs(current_dir_path)
    return current_dir_path