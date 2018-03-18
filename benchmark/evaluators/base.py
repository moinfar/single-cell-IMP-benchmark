import abc

import six


@six.add_metaclass(abc.ABCMeta)
class AbstractEvaluator:

    @abc.abstractmethod
    def prepare(self):
        """
        Prepare evaluator (i.g. downloading data sets, ...).
        :return: Returns None and raises exception in case of any problem.
        """
        pass

    @abc.abstractmethod
    def generate_test_bench(self, count_file_path, seed):
        """
        Generates a deterministic or probabilistic expression profile containing noise and dropout.
        :param count_file_path: The file, which expression profile should be stored in.
        :param seed: Random generator seed. To obtain reproducible results.
        :return: Returns a unique ID for this experiment.
        """
        pass

    @abc.abstractmethod
    def evaluate_result(self, uid, processed_count_file_path, result_file, seed):
        """
        Evaluates the result obtained from an algorithm.
        :param uid: unique ID returned by `generate_test_bench` function.
        :param processed_count_file_path: The result which should be evaluated.
        :param result_file: A file to store evaluation results and additional info into.
        :param seed: Random generator seed. To obtain reproducible results.
                     Note that evaluation may be probabilistic.
        :return: Returns a dictionary containing entries of the form "criteria": "value".
                 Note that evaluation results will also be saved in `result_file`
        """
        pass
