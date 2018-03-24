import os

import numpy as np

from evaluators.numerical import GridMaskedDataPredictionEvaluator
from framework.conf import settings
from utils.base import generate_seed
from utils.dataset import DataSet_10xPBMC4k

seed = generate_seed()

# eval1 = CellCyclePreservationEvaluator()
# eval1.prepare()
# uid = eval1.generate_test_bench(os.path.join(settings.IO_DIR, "some_file.txt"), seed=seed)
# results = eval1.evaluate_result(uid, os.path.join(settings.IO_DIR, "some_file.txt"),
#                                 os.path.join(settings.RESULTS_DIR, "result_%d.txt" % uid), seed=seed)
#
# pprint(results)


dataset2 = DataSet_10xPBMC4k()
dataset2.prepare()
print(dataset2.keys())
df = dataset2.get("GRCh38")
# 4k is too much
subset = np.random.choice(df.shape[1], 500, replace=False)
df = df.iloc[:, subset].copy()
eval2 = GridMaskedDataPredictionEvaluator(df)
eval2.prepare()
uid = eval2.generate_test_bench(os.path.join(settings.IO_DIR, "some_file.txt"), seed=seed)
results = eval2.evaluate_result(uid, os.path.join(settings.IO_DIR, "some_file.txt"),
                                os.path.join(settings.RESULTS_DIR, "result_%d.txt" % uid), seed=seed)
