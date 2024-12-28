import os

def current_dir_path_for(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)