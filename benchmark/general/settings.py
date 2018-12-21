import os


DEBUG = True

# Addresses
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(os.path.join(BASE_DIR, "files"))
STATIC_FILES_DIR = os.path.join(os.path.join(BASE_DIR, "static"))

CACHE_DIR = os.path.join(os.path.join(FILES_DIR, "cache"))
STORAGE_DIR = os.path.join(os.path.join(FILES_DIR, "storage"))

# Available data sets
data_sets = {
    'ERP006670': "utils.data_set.DataSet_ERP006670",
    'CELL_CYCLE': "utils.data_set.DataSet_ERP006670",
    'GSE60361': "utils.data_set.DataSet_GSE60361",
    'CORTEX_3005': "utils.data_set.DataSet_GSE60361",
    'SRP041736': "utils.data_set.DataSet_SRP041736",
    'POLLEN': "utils.data_set.DataSet_SRP041736",
    'SRP041736-HQ': "utils.data_set.DataSet_SRP041736_HQ",
    'POLLEN-HQ': "utils.data_set.DataSet_SRP041736_HQ",
    'SRP041736-LQ': "utils.data_set.DataSet_SRP041736_LQ",
    'POLLEN-LQ': "utils.data_set.DataSet_SRP041736_LQ",
    'GSE100866': "utils.data_set.DataSet_GSE100866",
    'CITE_Seq': "utils.data_set.DataSet_GSE100866",
    'CITE-CBMC': "utils.data_set.DataSet_GSE100866_CBMC",
    'CITE-PBMC': "utils.data_set.DataSet_GSE100866_PBMC",
    'CITE-CD8': "utils.data_set.DataSet_GSE100866_CD8",
    '10xPBMC4k': "utils.data_set.DataSet_10xPBMC4k"
}
