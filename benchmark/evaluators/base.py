import abc
import random as py_random

import six
from numpy import random as np_random


@six.add_metaclass(abc.ABCMeta)
class AbstractEvaluator:

    def __init__(self, uid):
        """
        :param uid: unique ID to identify current test.
        """
        self.uid = uid

    @abc.abstractmethod
    def prepare(self):
        """
        Prepare evaluator for test bench generation (i.g. downloading data sets, ...).
        :return: Returns None and raises exception in case of any problem.
        """
        pass

    @abc.abstractmethod
    def generate_test_bench(self, count_file_path, **kwargs):
        """
        Generates a deterministic or probabilistic expression profile containing noise and dropout.
        :param count_file_path: The file, which expression profile should be stored in.
        :param kwargs: Other evaluator specific arguments.
        :return: Returns None.
        """
        pass

    @abc.abstractmethod
    def evaluate_result(self, processed_count_file_path, result_prefix, **kwargs):
        """
        Evaluates the result obtained from an algorithm.
        :param processed_count_file_path: The result which should be evaluated.
        :param result_prefix: Prefix for files that evaluation results are stored in.
        :param kwargs: Other evaluator specific arguments.
        :return: Returns a dictionary containing entries of the form "criteria": "value".
                 Note that evaluation results will also be saved in `result_file`
        """
        pass

    @staticmethod
    def set_seed(seed):
        """
        Set seed for used random generators.
        :param seed: Random generator seed. To obtain reproducible results.
                     Note that both generation and evaluation may be probabilistic.
        :return: Returns None.
        """
        py_random.seed(seed)
        np_random.seed(seed)
