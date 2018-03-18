import os
from pprint import pprint

from evaluators.biological import CellCyclePreservationEvaluator
from framework.conf import settings
from framework.utils import generate_seed

seed = generate_seed()

eval1 = CellCyclePreservationEvaluator()
eval1.prepare()
uid = eval1.generate_test_bench(os.path.join(settings.IO_DIR, "some_file.txt"), seed=seed)
results = eval1.evaluate_result(uid, os.path.join(settings.IO_DIR, "some_file.txt"),
                                os.path.join(settings.RESULTS_DIR, "result_%d.txt" % uid), seed=seed)

pprint(results)
