import json
import os

import numpy as np

from evaluators.biological import CellCyclePreservationEvaluator
from evaluators.numerical import GridMaskedDataPredictionEvaluator, RandomPointMaskedDataPredictionEvaluator
from framework.conf import settings
from utils.base import load_class, make_sure_dir_exists, get_data_set_class


def list_benchmarks():
    print("<benchmark_id>\t<benchmark_name>")
    for i, evaluator_str in enumerate(settings.evaluators):
        evaluator = load_class(evaluator_str)
        print("%13d\t %s" % (i, evaluator.__name__))


def load_test_info_list():
    test_info_list = []
    make_sure_dir_exists(settings.TEST_INFO_DIR)
    test_info_filenames = os.listdir(settings.TEST_INFO_DIR)
    for test_info_filename in test_info_filenames:
        test_info_path = os.path.join(settings.TEST_INFO_DIR, test_info_filename)
        with open(test_info_path, 'r') as json_file:
            test_info = json.load(json_file)
        test_info_list.append(test_info)
    return test_info_list


def list_tests():
    test_info_list = load_test_info_list()
    print("<test_id>\t<benchmark_name>          \t<status>")
    for test_info in test_info_list:
        print("%8d\t%s\t%s" % (
            test_info['test_id'], test_info['evaluator'], test_info['status']
        ))
    pass


def list_data_sets():
    for data_set_name in settings.data_sets:
        data_set = get_data_set_class(data_set_name)()
        for key in data_set.keys():
            print("%s-%s" % (data_set_name, key))


def make_sure_test_exists(test_id):
    test_info_list = load_test_info_list()
    test_info = None
    for test_info in test_info_list:
        if test_info['test_id'] == test_id:
            break
    if test_info['test_id'] != test_id:
        print("No such test available.")
        exit(1)
    return test_info


def save_test_info(test_id, uid, count_path, result_path, evaluator, status):
    test_info = dict(
        uid=uid,
        test_id=test_id,
        count_file_path=count_path,
        evaluator=type(evaluator).__name__,
        status=status,
        result_path=result_path
    )
    test_info_path = os.path.join(settings.TEST_INFO_DIR, "%s.json" % uid)
    make_sure_dir_exists(os.path.dirname(test_info_path))
    with open(test_info_path, 'w') as json_file:
        json.dump(test_info, json_file)


def generate_cell_cycle(test_id, count_file_path):
    uid = "%d_cell_cycle" % test_id
    evaluator = CellCyclePreservationEvaluator(uid)
    evaluator.prepare()
    evaluator.generate_test_bench(count_file_path)
    return uid, evaluator


def evaluate_cell_cycle(uid, seed, is_normal, imputed_count_file_path, result_path):
    assert uid.endswith("_cell_cycle")
    evaluator = CellCyclePreservationEvaluator(uid, normalize_data_for_evaluation=not is_normal)
    evaluator.prepare()
    evaluator.set_seed(seed)
    results = evaluator.evaluate_result(imputed_count_file_path, result_path)
    return uid, evaluator, results


def generate_grid_prediction(test_id, count_file_path, data_set_name, seed, n_samples, grid_ratio):
    uid = "%d_grid_pattern" % test_id
    data_set_name, key = data_set_name.split("-")
    data_set = get_data_set_class(data_set_name)()
    data_set.prepare()
    data = data_set.get(key)
    if n_samples != "all":
        np.random.seed(0)
        n_samples = int(n_samples)
        subset = np.random.choice(data.shape[1], n_samples, replace=False)
        data = data.iloc[:, subset].copy()
    rows_ratio = float(grid_ratio.split("x")[0])
    columns_ratio = float(grid_ratio.split("x")[1])
    evaluator = GridMaskedDataPredictionEvaluator(uid,
                                                  reference_data_frame=data,
                                                  rows_ratio=rows_ratio,
                                                  columns_ratio=columns_ratio)
    evaluator.set_seed(seed)
    evaluator.prepare()
    evaluator.generate_test_bench(count_file_path)
    return uid, evaluator


def generate_random_prediction(test_id, count_file_path, data_set_name, seed, n_samples, dropout_count):
    uid = "%d_random_pattern" % test_id
    data_set_name, key = data_set_name.split("-")
    data_set = get_data_set_class(data_set_name)()
    data_set.prepare()
    data = data_set.get(key)
    dropout_count = int(dropout_count)
    if n_samples != "all":
        np.random.seed(0)
        n_samples = int(n_samples)
        subset = np.random.choice(data.shape[1], n_samples, replace=False)
        data = data.iloc[:, subset].copy()
    evaluator = RandomPointMaskedDataPredictionEvaluator(
        uid, reference_data_frame=data, dropout_count=dropout_count)
    evaluator.set_seed(seed)
    evaluator.prepare()
    evaluator.generate_test_bench(count_file_path)
    return uid, evaluator


def evaluate_grid_prediction(uid, imputed_count_file_path, result_path):
    assert uid.endswith("_grid_pattern")
    evaluator = GridMaskedDataPredictionEvaluator(uid, reference_data_frame=None)
    results = evaluator.evaluate_result(imputed_count_file_path, result_path)
    return uid, evaluator, results


def evaluate_random_prediction(uid, imputed_count_file_path, result_path):
    print(uid)
    assert uid.endswith("_random_pattern")
    evaluator = RandomPointMaskedDataPredictionEvaluator(uid, reference_data_frame=None)
    results = evaluator.evaluate_result(imputed_count_file_path, result_path)
    return uid, evaluator, results
