import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

# UPDATE ATTACK


def update_attack_on_marked_attr(init_data, x, marked_attribute, fraction, key, imputer = 0):

    # Calculate the number of cells to replace
    num_cells_to_replace = int(fraction * x.size)

    # Randomly choose indices of the numpy array
    np.random.seed(key)
    indices = np.random.choice(range(x.size), size=num_cells_to_replace, replace=False)


    z = init_data.copy()
    z[:, marked_attribute] = x.copy()
    z[indices, marked_attribute] = np.nan

    if imputer == 0:
        k = int(np.round(np.sqrt(z.shape[0])))
        k = k + (k % 2)
        knn_imputer = KNNImputer(n_neighbors=k)
        res = knn_imputer.fit_transform(z)
    elif imputer == 1:
        iterative_imputer = IterativeImputer(max_iter=1000)
        res = iterative_imputer.fit_transform(z)
    else:
        simple_imputer = SimpleImputer(strategy='mean')
        res = simple_imputer.fit_transform(z)

    return res[:, marked_attribute]


def insert_missing_values(df, fraction):
    # Calculate the total number of cells in the dataframe
    total_cells = df.shape[0] * df.shape[1]

    # Calculate the number of cells to replace
    num_cells_to_replace = int(fraction * total_cells)

    # Randomly choose indices of the numpy array
    indices = np.random.choice(range(df.size), size=num_cells_to_replace, replace=False)

    # Convert indices to 2D format
    row_indices, col_indices = np.unravel_index(indices, df.shape)

    # Replace the values at the selected indices
    for row, column in zip(row_indices, col_indices):
        df.iloc[row, column] = np.nan

    return df

def update_attack_on_whole_db(init_data, fraction, inputer = 0):
    X = insert_missing_values(init_data, fraction)
    if inputer == 0:
        k = int(np.round(np.sqrt(X.shape[0])))
        k = k + (k % 2 == 0)
        knn_imputer = KNNImputer(n_neighbors=k)
        df_knn = knn_imputer.fit_transform(X)
        return df_knn
    else:
        iterative_imputer = IterativeImputer(max_iter=1000)
        df_iterative = iterative_imputer.fit_transform(X)
        return df_iterative


def removal_attack(x, fraction, key):
    del_el = int(len(x) * fraction)
    np.random.seed(key)
    indeces = np.random.choice(len(x), del_el, replace=False)
    mask=np.full(len(x),True,dtype=bool)
    mask[indeces]=False
    res = x[mask]
    return res

def zero_out_attack(x, fraction, key):
    att_el = int(len(x) * fraction)
    np.random.seed(key)
    indeces = np.random.choice(len(x), att_el, replace=False)
    z = x.copy()
    z[indeces] = 0
    return z

def add_attack(init_data, x, marked_attribute, fraction, key, inputer = 0):
    nr_el = int(len(x) * fraction)

    np.random.seed(key)
    indeces = np.random.choice(len(x), nr_el)
    y = x.copy()
    new_entry = np.full(init_data.shape[1], np.nan)
    z = init_data.copy()
    z[:, marked_attribute] = y
    for i in indeces:
        z = np.insert(z, i, new_entry, axis=0)

    match inputer:
        case 0:
            k = int(np.round(np.sqrt(x.shape[0])))
            k = k + (k % 2 == 0)
            knn_imputer = KNNImputer(n_neighbors=k)
            df_knn = knn_imputer.fit_transform(z)
            y = df_knn[:, marked_attribute]
        case _:
            iterative_imputer = IterativeImputer(max_iter=1000)
            df_iterative = iterative_imputer.fit_transform(z)
            y = df_iterative[:, marked_attribute]
    # print(np.mean(x), np.mean(y))
    return y




