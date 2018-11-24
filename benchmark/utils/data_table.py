import pandas as pd
import numpy as np

from utils.base import load_gzip_pickle


def shuffle_and_rename_columns(data, prefix="cell", disabled=False):
    original_columns = data.columns.values

    if not disabled:
        column_permutation = np.random.permutation(range(len(data.columns.values)))
    else:
        column_permutation = range(len(data.columns.values))
    permuted_data = data[data.columns.values[column_permutation]]

    if not disabled:
        permuted_data.columns = ["%s_%d" % (prefix, i) for i in range(len(original_columns))]

    return permuted_data, original_columns, column_permutation


def rearrange_and_rename_columns(data, original_columns, column_permutation):
    reverse_permutation = np.argsort(column_permutation)

    rearranged_data = data[data.columns.values[reverse_permutation]]
    rearranged_data.columns = original_columns

    return rearranged_data


def read_csv(filename, index_col=0, header=0):
    return pd.read_csv(filename, sep=None, index_col=index_col, header=header, engine="python")


def write_csv(data, filename):
    if filename.endswith(".csv"):
        data.to_csv(filename, sep=",", index_label="")
    elif filename.endswith(".tsv"):
        data.to_csv(filename, sep="\t", index_label="")
    elif filename.endswith(".csv.gz"):
        data.to_csv(filename, sep=",", index_label="", compression="gzip")
    elif filename.endswith(".tsv.gz"):
        data.to_csv(filename, sep="\t", index_label="", compression="gzip")
    else:
        raise NotImplementedError("Unrecognized format for file %s" % filename)


def read_table_file(filename):
    if filename.endswith(".csv") or filename.endswith(".tsv") or \
            filename.endswith(".csv.gz") or filename.endswith(".tsv.gz"):
        return read_csv(filename)
    elif filename.endswith(".pkl.gz"):
        return load_gzip_pickle(filename)
    else:
        raise NotImplementedError("Unrecognized format for file %s" % filename)
