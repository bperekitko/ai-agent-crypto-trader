import json
import os
import pickle
import pandas as pd

# Displaying all columns when printing
pd.set_option('display.max_columns', None)

# Disable line wrap when printing
pd.set_option('display.expand_frame_repr', False)

__CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def load_scaler(name):
    return __deserialize(os.path.join(__CURRENT_DIR_PATH, 'scalers', name))


def save_scaler(name, scaler):
    __serialize(os.path.join(__CURRENT_DIR_PATH, 'scalers', name), scaler)
    return name


def save_box_cox_lambda(name, fitted_lambda):
    __serialize(os.path.join(__CURRENT_DIR_PATH, 'box_cox_lambdas', name), fitted_lambda)
    return name


def load_box_cox_lambda(name):
    return __deserialize(os.path.join(__CURRENT_DIR_PATH, 'box_cox_lambdas', name))


def save_to_json(name, data):
    with open(os.path.join(__CURRENT_DIR_PATH, f'{name}.json'), 'w') as file:
        json.dump(data, file, indent=4)
    return name


def load_from_json(name):
    with open(os.path.join(__CURRENT_DIR_PATH, f'{name}.json', 'r')) as file:
        return json.load(file)


def __deserialize(path):
    with open(f'{path}.pkl', 'rb') as file:
        return pickle.load(file)


def __serialize(path, data):
    with open(f'{path}.pkl', 'wb') as file:
        pickle.dump(data, file)
