import abc
import csv
import os

import numpy as np
import pandas as pd
import six
from scipy.io import mmread

from framework.conf import settings
from utils.base import make_sure_dir_exists, download_file_if_not_exists, extract_compressed_file, load_class
from utils.data_table import read_csv


def get_data_set_class(dataset_id):
    return load_class(settings.data_sets[dataset_id])


@six.add_metaclass(abc.ABCMeta)
class DataSet:

    @abc.abstractmethod
    def prepare(self):
        pass

    @abc.abstractmethod
    def keys(self):
        pass

    @abc.abstractmethod
    def get(self, key):
        pass


class DataSet_ERP006670(DataSet):
    DATA_SET_URL = "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-2805/E-MTAB-2805.processed.1.zip"
    DATA_SET_MD5_SUM = "6e9f2611d670e14bb0fe750682843e10"

    def __init__(self):
        self.DATA_SET_URL = DataSet_ERP006670.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_ERP006670.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join("cell_cycle", os.path.basename(self.DATA_SET_URL))
        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR, self.DATA_SET_FILE_PATH)
        self.DATA_SET_DIR = "%s.d" % self.DATA_SET_FILE_PATH

        self.G1_DATA_PATH = os.path.join(self.DATA_SET_DIR, "G1_singlecells_counts.txt")
        self.G2M_DATA_PATH = os.path.join(self.DATA_SET_DIR, "G2M_singlecells_counts.txt")
        self.S_DATA_PATH = os.path.join(self.DATA_SET_DIR, "S_singlecells_counts.txt")

        self.KEYS = ["G1", "G2M", "S"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def _extract_data_set(self):
        extract_compressed_file(self.DATA_SET_FILE_PATH, self.DATA_SET_DIR)

    def prepare(self):
        self._download_data_set()
        self._extract_data_set()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        key_to_data_path_mapping = {
            "G1": self.G1_DATA_PATH,
            "G2M": self.G2M_DATA_PATH,
            "S": self.S_DATA_PATH
        }

        data = pd.read_csv(key_to_data_path_mapping[key], sep="\t")
        return data


class DataSet_10xPBMC4k(DataSet):
    DATA_SET_URL = "http://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc4k/" \
                   "pbmc4k_filtered_gene_bc_matrices.tar.gz"
    DATA_SET_MD5_SUM = "f61f4deca423ef0fa82d63fdfa0497f7"

    def __init__(self):
        self.DATA_SET_URL = DataSet_10xPBMC4k.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_10xPBMC4k.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join("PBMC4k", os.path.basename(self.DATA_SET_URL))
        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR, self.DATA_SET_FILE_PATH)
        self.DATA_SET_DIR = "%s.d" % self.DATA_SET_FILE_PATH

        genome = "GRCh38"
        files_dir = os.path.join(self.DATA_SET_DIR, "filtered_gene_bc_matrices", genome)
        self.BARCODE_DATA_PATH = os.path.join(files_dir, "barcodes.tsv")
        self.GENE_DATA_PATH = os.path.join(files_dir, "genes.tsv")
        self.MATRIX_DATA_PATH = os.path.join(files_dir, "matrix.mtx")

        self.KEYS = [genome, "data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def _extract_data_set(self):
        extract_compressed_file(self.DATA_SET_FILE_PATH, self.DATA_SET_DIR)

    def prepare(self):
        self._download_data_set()
        self._extract_data_set()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        barcode_data = [row for row in csv.reader(open(self.BARCODE_DATA_PATH), delimiter="\t")]
        barcodes = [row[0] for row in barcode_data]

        gene_data = [row for row in csv.reader(open(self.GENE_DATA_PATH), delimiter="\t")]
        gene_ids = [row[0] for row in gene_data]
        # gene_names = [row[1] for row in gene_data]

        matrix = mmread(self.MATRIX_DATA_PATH).todense()

        data = pd.DataFrame(matrix, index=gene_ids, columns=barcodes)
        data.index.name = 'Symbol'

        return data


class DataSet_GSE60361(DataSet):
    # DATA_SET_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/" \
    #                "GSE60361/suppl/GSE60361_C1-3005-Expression.txt.gz"
    # DATA_SET_MD5_SUM = "fbf6f0ec39d54d8aac7233c56d0c9e30"
    DATA_SET_URL = "https://storage.googleapis.com/linnarsson-lab-www-blobs/" \
                   "blobs/cortex/expression_mRNA_17-Aug-2014.txt"
    DATA_SET_MD5_SUM = "6bb4c2a9ade87b16909d39004021849e"

    def __init__(self):
        self.DATA_SET_URL = DataSet_GSE60361.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_GSE60361.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join("GSE60361", os.path.basename(self.DATA_SET_URL))
        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR, self.DATA_SET_FILE_PATH)

        self.KEYS = ["mRNA", "details", "data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def prepare(self):
        self._download_data_set()

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        if key == "mRNA" or key == "data":
            full_data = read_csv(self.DATA_SET_FILE_PATH, header=None)
            data = full_data.iloc[11:-1, 1:3006]
            data.columns = full_data.iloc[7, 1:3006]
            data.index.name = ""
            data.columns.name = ""
            data = data.astype("int64")
            return data
        elif key == "details":
            full_data = read_csv(self.DATA_SET_FILE_PATH, header=None)
            data = full_data.iloc[:10, 1:3006]
            data.columns = full_data.iloc[7, 1:3006]
            data.index = full_data.iloc[:10, 0].values
            data.index.name = ""
            data.columns.name = ""
            return data


class DataSet_SRP041736(DataSet):
    DATA_SET_URL = "http://duffel.rail.bio/recount/v2/SRP041736/counts_gene.tsv.gz"
    DATA_SET_MD5_SUM = "535271b8cd81a93eb210254b766ebcbb"

    def __init__(self):
        self.DATA_SET_URL = DataSet_SRP041736.DATA_SET_URL
        self.DATA_SET_MD5_SUM = DataSet_SRP041736.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR,
                                               "DataSet_SRP041736",
                                               os.path.basename(self.DATA_SET_URL))

        self.EXPERIMENT_INFO_FILE_PATH = os.path.join(settings.STATIC_FILES_DIR,
                                                      "data", "SRP041736_info.txt")

        self.KEYS = ["details", "HQ-details", "LQ-details", "data", "HQ-data", "LQ-data"]

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def prepare(self):
        self._download_data_set()
        assert os.path.exists(self.EXPERIMENT_INFO_FILE_PATH)

    def keys(self):
        return self.KEYS

    def get(self, key):
        assert key in self.keys()

        info = pd.read_csv(self.EXPERIMENT_INFO_FILE_PATH, engine="python", sep="\t", comment="#")
        info = info[np.logical_and(info["Experiment"] != "SRX534506", info["Experiment"] != "SRX534553")]
        info = info.sort_values(by=["Experiment", "MBases"])
        info.index = info["Run"].values
        info["class"] = [sample_name.split("_")[0] for sample_name in info["Sample_Name"]]
        info = info.transpose()

        if key == "details":
            return info

        if key == "HQ-details":
            return info.iloc[:, range(1, 692, 2)]

        if key == "LQ-details":
            return info.iloc[:, range(0, 692, 2)]

        data = pd.read_csv(self.DATA_SET_FILE_PATH, engine="python", sep=None, index_col=-1)
        data = data.astype("int64")
        data["pure_gene_id"] = [gene_name.split(".")[0] for gene_name in list(data.index.values)]
        data = data.groupby(["pure_gene_id"]).sum()
        data.index.name = "gene_id"
        data = data[info.loc["Run"].values]

        if key == "data":
            return data

        if key == "HQ-data":
            return data.iloc[:, range(1, 692, 2)]

        if key == "LQ-data":
            return data.iloc[:, range(0, 692, 2)]


class DataSet_SRP041736_HQ(DataSet):
    def __init__(self):
        self.ds = DataSet_SRP041736()
        self.KEYS = ["details", "data"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "details":
            return self.ds.get("HQ-details")

        if key == "data":
            return self.ds.get("HQ-data")


class DataSet_SRP041736_LQ(DataSet):
    def __init__(self):
        self.ds = DataSet_SRP041736()
        self.KEYS = ["details", "data"]

    def prepare(self):
        self.ds.prepare()

    def keys(self):
        return self.KEYS

    def get(self, key):
        if key == "details":
            return self.ds.get("LQ-details")

        if key == "data":
            return self.ds.get("LQ-data")
