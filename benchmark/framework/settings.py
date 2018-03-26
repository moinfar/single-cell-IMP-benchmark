import os

DEBUG = True

# Addresses
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(os.path.join(BASE_DIR, "files"))

CACHE_DIR = os.path.join(os.path.join(FILES_DIR, "cache"))
STORAGE_DIR = os.path.join(os.path.join(FILES_DIR, "storage"))
TEST_INFO_DIR = os.path.join(os.path.join(FILES_DIR, "test_info"))

# Available evaluators
evaluators = [
    'evaluators.biological.CellCyclePreservationEvaluator',
    'evaluators.numerical.GridMaskedDataPredictionEvaluator',
]

# Available data sets
data_sets = {
    'ERP006670': "utils.dataset.DataSet_ERP006670",
    '10xPBMC4k': "utils.dataset.DataSet_10xPBMC4k"
}
