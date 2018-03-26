#!python

"""
deepIMP-benchmark
A benchmarking suite to evaluate single-cell RNA-seq imputation algorithms.

Usage:
  run.py list benchmarks
  run.py list tests
  run.py list datasets
  run.py generate cell-cycle <test_id> --output=COUNT_FILE [options]
  run.py evaluate cell-cycle <test_id> --input=IMPUTED_FILE --result=RESULT_FILE [--is-normal] [options]
  run.py generate grid-prediction <test_id> --output=COUNT_FILE [--dataset=DATASET] [--n-samples=N_SAMPLES] [--grid-ratio=GRID_RATIO] [options]
  run.py evaluate grid-prediction <test_id> --input=IMPUTED_FILE --result=RESULT_FILE [options]
  run.py (-h | --help)
  run.py --version

-o, --output=COUNT_FILE        Address where count matrix will be stored in
-i, --input=IMPUTED_FILE       Address of file containing imputed count matrix
-r, --result=RESULT_FILE       Address where evaluation result will be stored in

--is-normal                    Evaluator will not normalize the input if this flag is set
-d, --dataset=DATASET          Dataset to be used in a benchmark [default: 10xPBMC4k-GRCh38]
-n, --n-samples=N_SAMPLES      Number of samples used from dataset (e.g. 500) [default: all]
-g, --grid-ratio=GRID_RATIO    Ratio of the grid uses in evaluator[default: 0.2x0.2]

Options:
  -S, --seed=n                 Seed for random generator (random if not provided)
  -D, --debug                  Prints debugging info.
"""

from __future__ import print_function

import os

from docopt import docopt

from framework.conf import settings
from main import list_benchmarks, list_tests, list_data_sets, generate_cell_cycle, generate_grid_prediction, \
    save_test_info, make_sure_test_exists, evaluate_cell_cycle, evaluate_grid_prediction
from utils.base import generate_seed

if __name__ == '__main__':
    arguments = docopt(__doc__, version='deepIMP-benchmark 0.1')

    settings.DEBUG = arguments['--debug']
    seed = int(arguments['--seed']) if arguments['--seed'] else generate_seed()

    if arguments['list']:
        if arguments['benchmarks']:
            list_benchmarks()
        elif arguments['tests']:
            list_tests()
        elif arguments['datasets']:
            list_data_sets()
    elif arguments['generate']:
        test_id = int(arguments['<test_id>'])
        count_file_path = os.path.abspath(arguments['--output'])
        if arguments['cell-cycle']:
            uid, evaluator = generate_cell_cycle(test_id, count_file_path)
        elif arguments['grid-prediction']:
            data_set_name = arguments['--dataset']
            n_samples = arguments['--n-samples']
            grid_ratio = arguments['--grid-ratio']
            uid, evaluator = generate_grid_prediction(test_id, count_file_path,
                                                      data_set_name, seed, n_samples, grid_ratio)
        else:
            raise ModuleNotFoundError
        save_test_info(test_id, uid, count_file_path, None,
                       evaluator, "generated")
        print("Count file saved to `%s`" % count_file_path)
    elif arguments['evaluate']:
        test_id = int(arguments['<test_id>'])
        imputed_count_file_path = os.path.abspath(arguments['--input'])
        result_path = os.path.abspath(arguments['--result'])
        test_info = make_sure_test_exists(test_id)
        if arguments['cell-cycle']:
            is_normal = arguments['--is-normal']
            uid, evaluator, results = evaluate_cell_cycle(test_info['uid'], seed,
                                                          is_normal, imputed_count_file_path, result_path)
        elif arguments['grid-prediction']:
            is_normal = arguments['--is-normal']
            uid, evaluator, results = evaluate_grid_prediction(test_info['uid'],
                                                               imputed_count_file_path, result_path)
        else:
            raise ModuleNotFoundError
        save_test_info(test_id, uid, imputed_count_file_path, result_path,
                       evaluator, "evaluated")
        for metric in results:
            print("%s: %4f" % (metric, results[metric]))
        print("Results saved to `%s`" % result_path)
