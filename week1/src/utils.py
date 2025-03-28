import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return shuffle(data, random_state=42)


def unique_values(data, column):
    return data[column].unique()


def split_dataset(data, column, value):
    return data[data[column] == value], data[data[column] != value]


def most_common_label(data, target):
    return data[target].mode()[0]


def entropy(data, target):
    total = len(data)
    proportions = data[target].value_counts() / total
    return -sum(proportions * np.log2(proportions))
