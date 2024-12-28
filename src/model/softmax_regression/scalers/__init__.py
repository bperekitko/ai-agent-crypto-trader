import pickle
import os

__CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

def load_scaler(name):
    with open(os.path.join(__CURRENT_DIR_PATH, f'{name}.pkl', 'rb')) as file:
        return pickle.load(file)

def save_scaler(name, scaler):
    with open(os.path.join(__CURRENT_DIR_PATH, f'{name}.pkl'), 'wb') as file:
        pickle.dump(scaler, file)
    return name