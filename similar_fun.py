"This module is for similar_dict function."
import numpy as np
import pandas as pd
import re


def price_class(value: float, borders: list):
    """Determine price class according to the prices.
        Parameters:
                    value: price must be defined in the price class
                    borders: prices are the boundaries of the price class
        Returns a column with price classes.
    """
    n = len(borders)
    if value < borders[0]:
        return 0 
    elif value >= borders[n-1]:
        return n
    else:
        for i in range(1, n):
            if (value >= borders[i-1]) and (value < borders[i]):
                return i


def dummies_encoding(selected: pd.DataFrame, selected_cols: pd.Index, weights: np.array):
    """Encodes the main features into dummies.
        Parameters:
                    selected: a dataframe with the main features
                    selected_cols: names of the main features
                    weights: an array with weights of the main features  
        Returns a dataframe with the main encoded features.
    """
    features = pd.DataFrame({})
    for i in range(len(selected_cols)):
        coef = round(weights[i])
        cat_cols = coef * pd.get_dummies(selected[selected_cols[i]])
        features = pd.concat([features, cat_cols], axis=1)
    return features


def search_for_neighbors(features: pd.DataFrame):
    """Looks for neighbors.
        Parameters:
                    features: a dataframe with the main encoded features 
        Returns an array of neighbor indexes.
    """
    x = features.values
    distances = euclidean_distance(x, x)
    available = distances.shape[1]
    diag_index = np.arange(available)
    distances[diag_index, diag_index] = np.inf ### exclude the same object
    k = min(50, available)
    closest_ind = k_nearest_neighbors(distances, k)
    return closest_ind


def sorted_nearest_indexes(df: pd.DataFrame, brand_col: str, indexes_array: np.array, apple=False):
    """Sortes neighbors by price and brand.
        Parameters:
                    df: dataframe to be sorted 
                    brand_col: a price column name to sort
                    indexes_array: an array of neighbor indexes to be sorted 
                    apple: a bool variable responds for sorting by brand 
        Returns a sorted array of neighbor indexes.
    """
    indexes = []
    for i in range(df.shape[0]):
        neighbors_indexes = indexes_array[i, :]
        diff_prices = np.argsort(np.abs(df.Price.iloc[i] - df.Price.iloc[neighbors_indexes].values))
        neighbors_indexes = neighbors_indexes[diff_prices]
        if not apple: 
            brands = df.iloc[neighbors_indexes, :][brand_col].values 
            same_brand = np.where(brands == df.iloc[i, :][brand_col])[0]
            sorted_indexes = list(neighbors_indexes[same_brand])
            neighbors_indexes = list(neighbors_indexes)
            for item in sorted_indexes:
                neighbors_indexes.remove(item)
            sorted_indexes.extend(neighbors_indexes)
        else:
            sorted_indexes = list(neighbors_indexes)
        indexes.append(sorted_indexes)
    return np.array(indexes)


def convert_to_float(val: str) -> float:
    """
    Returns a column that indicates whether the numeric value is valid.
    """
    try:
        num = float(val)
        return 1
    except ValueError:
        return 0
