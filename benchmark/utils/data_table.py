import numpy as np


def shuffle_and_rename_columns(data, prefix="cell"):
    original_columns = data.columns.values

    column_permutation = np.random.permutation(range(len(data.columns.values)))
    permuted_data = data[data.columns.values[column_permutation]]

    permuted_data.columns = ["%s_%d" % (prefix, i) for i in range(len(original_columns))]

    return permuted_data, original_columns, column_permutation


def rearrange_and_rename_columns(data, original_columns, column_permutation):
    reverse_permutation = np.argsort(column_permutation)

    rearranged_data = data[data.columns.values[reverse_permutation]]
    rearranged_data.columns = original_columns

    return rearranged_data
