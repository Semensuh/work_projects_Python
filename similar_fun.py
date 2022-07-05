"This module is for similar_dict function."
import numpy as np
import pandas as pd
import re


def encode_features(col, values):
    """
    Encodes according to sorted values.
    """
    for i in range(len(values)):
        if col == values[i]:
            return i

def price_class(value, borders):
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


def dummies_encoding(selected, selected_cols, weights):
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
        cat_cols = coef*pd.get_dummies(selected[selected_cols[i]])
        features = pd.concat([features, cat_cols], axis=1)
    return features


def euclidean_distance(x, y):
    """Returns the distance matrix between objects with features in the matrix x and y.
        Parameters:
                    x: an array of numeric features of all products
                    y: an array of numeric features of stock product.
    """
    sqr_x = np.array([np.sum(x * x, axis=1)])
    sqr_y = np.array([np.sum(y * y, axis=1)])
    x_y = np.dot(x,y.T)
    dist_x_y = np.where(sqr_x.T - 2 * x_y + sqr_y >= 0, sqr_x.T - 2 * x_y + sqr_y, 0) ### in case of an error of the form -4e-15
    return np.sqrt(dist_x_y)


def k_nearest_neighbors(distances, k):
    """Returns the first k indexes of nearest neighbors. 
        Parameters:
                    distances: the distance matrix of objects
                    k: a number of nearest neighbors to be found.
    """
    if 2*k < distances.shape[1]:
        k_sorted_index = np.argpartition(distances, k, axis = 1)
        k_sorted_dist = np.take_along_axis(distances, k_sorted_index, axis = 1)
        dist_k = np.delete(k_sorted_dist, np.s_[k:], axis=1)
        index_k = np.delete(k_sorted_index, np.s_[k:], axis=1)
        sorted_indexes = np.argsort(dist_k, axis=1)
        dist_ind = np.take_along_axis(index_k, sorted_indexes, axis=1)
    else:
        dist_ind = np.argsort(distances, axis=1)[:, :k]
    return dist_ind


def search_for_neighbors(features):
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


def sorted_nearest_indexes(df, brand_col, indexes_array, apple=False):
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


### Data preprocessing

def code_median(data, cat_feature, real_feature):
    "Encodes categorical feature by price."
    
    return (data[cat_feature].map(data.groupby(cat_feature)[real_feature].median()))


def convert_to_float(val):
    """
    Returns a column that indicates whether the numeric value is valid.
    """
    try:
        num = float(val)
        return 1
    except ValueError:
        return 0
