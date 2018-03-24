import abc
import csv
import os

import pandas as pd
import six
from scipy.io import mmread

from framework.conf import settings
from utils.base import make_sure_dir_exists, download_file_if_not_exists, extract_compressed_file


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

        self.KEYS = [genome]

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
        # gene_ids = [row[0] for row in gene_data]
        gene_names = [row[1] for row in gene_data]

        matrix = mmread(self.MATRIX_DATA_PATH).todense()

        data = pd.DataFrame(matrix, index=gene_names, columns=barcodes)
        data.index.name = 'GeneName'

        return data
