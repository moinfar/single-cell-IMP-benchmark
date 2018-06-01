import json
import os

import numpy as np
import pandas as pd

from evaluators.base import AbstractEvaluator
from framework import settings
from utils.base import make_sure_dir_exists, log


class GridMaskedDataPredictionEvaluator(AbstractEvaluator):
    def __init__(self, uid, reference_data_frame, rows_ratio=0.2, columns_ratio=0.2):
        super(GridMaskedDataPredictionEvaluator, self).__init__(uid)

        if reference_data_frame is not None:
            assert type(reference_data_frame) == pd.DataFrame

            self.data = reference_data_frame.copy()
            self.n_grid_rows = int(self.data.shape[0] * rows_ratio)
            self.n_grid_columns = int(self.data.shape[1] * columns_ratio)

    def prepare(self):
        self.data.columns = ["column_%d" % i for i in range(len(self.data.columns))]

    def generate_test_bench(self, count_file_path):
        count_file_path = os.path.abspath(count_file_path)

        # Generate elimination mask
        grid_rows = np.random.choice(self.data.shape[0], self.n_grid_rows, replace=False)
        grid_columns = np.random.choice(self.data.shape[1], self.n_grid_columns, replace=False)
        mask = pd.DataFrame(np.zeros_like(self.data),
                            index=self.data.index,
                            columns=self.data.columns)
        mask.iloc[grid_rows, grid_columns] = 1

        # Elimination
        low_quality_data = self.data * (1 - mask)

        # Shuffle new data
        column_permutation = np.random.permutation(self.data.shape[1])
        low_quality_data = low_quality_data.iloc[:, column_permutation]

        # Rename columns
        low_quality_data.columns = ["sample_%d" % i for i in range(len(low_quality_data.columns))]

        # Save hidden state
        hidden_state = {
            'grid_rows': [int(num) for num in grid_rows],
            'grid_columns': [int(num) for num in grid_columns],
            'column_permutation': [int(num) for num in column_permutation]
        }
        hidden_state_file_path = os.path.join(settings.STORAGE_DIR, "%s.json" % self.uid)
        make_sure_dir_exists(os.path.dirname(hidden_state_file_path))
        with open(hidden_state_file_path, 'w') as json_file:
            json.dump(hidden_state, json_file)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.pkl.xz" % self.uid)
        make_sure_dir_exists(os.path.dirname(hidden_data_file_path))
        self.data.to_sparse(fill_value=0).to_pickle(hidden_data_file_path, compression='xz')
        log("Benchmark hidden data saved to `%s` and\n"
            "                               `%s`" % (hidden_state_file_path, hidden_data_file_path))

        # Save test bench count file
        make_sure_dir_exists(os.path.dirname(count_file_path))
        low_quality_data.to_csv(count_file_path, sep=",", index_label="")
        log("Count file saved to `%s`" % count_file_path)

    def evaluate_result(self, processed_count_file_path, result_file):
        # Load hidden state and data
        hidden_state_file_path = os.path.join(settings.STORAGE_DIR, "%s.json" % self.uid)
        with open(hidden_state_file_path, 'r') as json_file:
            hidden_state = json.load(json_file)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.pkl.xz" % self.uid)
        data = pd.read_pickle(hidden_data_file_path, compression='xz').to_dense()

        grid_rows = hidden_state['grid_rows']
        grid_columns = hidden_state['grid_columns']
        column_permutation = hidden_state['column_permutation']

        # Load imputed data
        imputed_data = pd.read_csv(processed_count_file_path, sep=",", index_col=0)

        # Restore column order
        imputed_data = imputed_data.iloc[:, np.argsort(column_permutation)]
        imputed_data.columns = data.columns

        # Generate mask
        mask = pd.DataFrame(np.zeros_like(data),
                            index=data.index,
                            columns=data.columns)
        mask.iloc[grid_rows, grid_columns] = 1

        # Evaluation
        diff = np.abs(np.log(1 + data) - np.log(1 + imputed_data))
        mse_loss = float(np.sum(np.sum(mask * np.where(data != 0, 1, 0) * np.square(diff))) /
                         np.sum(np.sum(mask * np.where(data != 0, 1, 0))))

        metric_results = {
            'MSE': mse_loss
        }

        # Save results to a file
        make_sure_dir_exists(os.path.dirname(result_file))
        with open(result_file, 'w') as file:
            file.write("## METRICS:\n")
            for metric in metric_results:
                file.write("%s: %4f\n" % (metric, metric_results[metric]))

            file.write("\n## ADDITIONAL INFO:\n")
            file.write("# GOT (masked grid only):\n")
            grid = (imputed_data * np.where(data != 0, 1, np.NAN)).iloc[grid_rows, grid_columns]
            file.write(grid.to_string() + "\n")
            file.write("# GOLD STANDARD (masked grid only):\n")
            grid = (data * np.where(data != 0, 1, np.NAN)).iloc[grid_rows, grid_columns]
            file.write(grid.to_string() + "\n")

        log("Evaluation results saved to `%s`" % result_file)

        return metric_results


class RandomPointMaskedDataPredictionEvaluator(AbstractEvaluator):
    def __init__(self, uid, reference_data_frame, dropout_count=100):
        super(RandomPointMaskedDataPredictionEvaluator, self).__init__(uid)

        self.dropout_count = dropout_count

        if reference_data_frame is not None:
            assert type(reference_data_frame) == pd.DataFrame

            self.data = reference_data_frame.copy()

    def prepare(self):
        self.data.columns = ["column_%d" % i for i in range(len(self.data.columns))]

    def generate_test_bench(self, count_file_path):
        count_file_path = os.path.abspath(count_file_path)

        # Generate elimination mask
        non_zero_locations = []

        tmp_values = self.data.values
        for x in range(self.data.shape[0]):
            for y in range(self.data.shape[1]):
                if tmp_values[x, y] > 0:
                    non_zero_locations.append((x, y))
        del tmp_values

        mask = np.zeros_like(self.data)

        masked_locations = [non_zero_locations[index] for index in
                            np.random.choice(len(non_zero_locations),
                                             self.dropout_count, replace=False)]

        for (x, y) in masked_locations:
            mask[x, y] = 1

        mask = pd.DataFrame(mask, index=self.data.index, columns=self.data.columns)

        # Elimination
        low_quality_data = self.data * (1 - mask)

        # Shuffle new data
        column_permutation = np.random.permutation(self.data.shape[1])
        low_quality_data = low_quality_data.iloc[:, column_permutation]

        # Rename columns
        low_quality_data.columns = ["sample_%d" % i for i in range(len(low_quality_data.columns))]

        # Save hidden state
        hidden_state = {
            'masked_locations': masked_locations,
            'column_permutation': [int(num) for num in column_permutation]
        }
        hidden_state_file_path = os.path.join(settings.STORAGE_DIR, "%s.json" % self.uid)
        make_sure_dir_exists(os.path.dirname(hidden_state_file_path))
        with open(hidden_state_file_path, 'w') as json_file:
            json.dump(hidden_state, json_file)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.pkl.xz" % self.uid)
        make_sure_dir_exists(os.path.dirname(hidden_data_file_path))
        self.data.to_sparse(fill_value=0).to_pickle(hidden_data_file_path, compression='xz')
        log("Benchmark hidden data saved to `%s` and\n"
            "                               `%s`" % (hidden_state_file_path, hidden_data_file_path))

        # Save test bench count file
        make_sure_dir_exists(os.path.dirname(count_file_path))
        low_quality_data.to_csv(count_file_path, sep=",", index_label="")
        log("Count file saved to `%s`" % count_file_path)

    def evaluate_result(self, processed_count_file_path, result_file):
        # Load hidden state and data
        hidden_state_file_path = os.path.join(settings.STORAGE_DIR, "%s.json" % self.uid)
        with open(hidden_state_file_path, 'r') as json_file:
            hidden_state = json.load(json_file)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.pkl.xz" % self.uid)
        data = pd.read_pickle(hidden_data_file_path, compression='xz').to_dense()

        masked_locations = hidden_state['masked_locations']
        column_permutation = hidden_state['column_permutation']

        # Load imputed data
        imputed_data = pd.read_csv(processed_count_file_path, sep=",", index_col=0)

        # Restore column order
        imputed_data = imputed_data.iloc[:, np.argsort(column_permutation)]
        imputed_data.columns = data.columns

        # Generate mask
        mask = np.zeros_like(data)
        for (x, y) in masked_locations:
            mask[x, y] = 1
        mask = pd.DataFrame(mask, index=data.index, columns=data.columns)

        # Evaluation
        diff = np.abs(np.log(1 + data) - np.log(1 + imputed_data))
        mse_loss = float(np.sum(np.sum(mask * np.where(data != 0, 1, 0) * np.square(diff))) /
                         np.sum(np.sum(mask * np.where(data != 0, 1, 0))))

        metric_results = {
            'MSE': mse_loss
        }

        # Save results to a file
        make_sure_dir_exists(os.path.dirname(result_file))
        with open(result_file, 'w') as file:
            file.write("## METRICS:\n")
            for metric in metric_results:
                file.write("%s: %4f\n" % (metric, metric_results[metric]))

            file.write("\n## ADDITIONAL INFO:\n")
            file.write("# GENE, CELL, GOLD_STANDARD, RESULT:\n")
            for (x, y) in masked_locations:
                file.write("%s\t%s\t%f\t%f\n" % (data.index.values[x],
                                                 data.columns.values[y],
                                                 data.iloc[x, y],
                                                 imputed_data.iloc[x, y]))

        log("Evaluation results saved to `%s`" % result_file)

        return metric_results
