import os
import pickle
import json
from urllib.request import urlretrieve


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def ensure_file(file_name, url):
    if not os.path.isfile(file_name):
        urlretrieve(url, file_name)


def save_obj(f_name, obj):
    with open(f_name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)


def read_json(f_name):
    with open(f_name, 'r') as f:
        return json.load(f)
