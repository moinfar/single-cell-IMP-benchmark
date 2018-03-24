import json
import os

import numpy as np
import pandas as pd

from evaluators.base import AbstractEvaluator
from framework import settings
from utils.base import make_sure_dir_exists, log


class GridMaskedDataPredictionEvaluator(AbstractEvaluator):
    def __init__(self, reference_data_frame, rows_ratio=0.2, columns_ratio=0.2):
        assert type(reference_data_frame) == pd.DataFrame

        self.data = reference_data_frame.copy()
        self.n_grid_rows = int(self.data.shape[0] * rows_ratio)
        self.n_grid_columns = int(self.data.shape[1] * columns_ratio)

    def prepare(self):
        self.data.index.name = 'Symbol'

    def generate_test_bench(self, uid, count_file_path):
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
        hidden_state_file_path = os.path.join(settings.STORAGE_DIR, "%d.json" % uid)
        make_sure_dir_exists(os.path.dirname(hidden_state_file_path))
        with open(hidden_state_file_path, 'w') as json_file:
            json.dump(hidden_state, json_file)
        log("Benchmark hidden data saved to `%s`" % hidden_state_file_path)

        # Save test bench count file
        make_sure_dir_exists(os.path.dirname(count_file_path))
        low_quality_data.to_csv(count_file_path, sep="\t")
        log("Count file saved to `%s`" % count_file_path)

        return uid

    def evaluate_result(self, uid, processed_count_file_path, result_file):
        # Load hidden state
        hidden_state_file_path = os.path.join(settings.STORAGE_DIR, "%d.json" % uid)
        with open(hidden_state_file_path, 'r') as json_file:
            hidden_state = json.load(json_file)

        grid_rows = hidden_state['grid_rows']
        grid_columns = hidden_state['grid_columns']
        column_permutation = hidden_state['column_permutation']

        # Load imputed data
        imputed_data = pd.read_csv(processed_count_file_path, sep="\t")
        imputed_data = imputed_data.set_index('Symbol')

        # Restore column order
        imputed_data = imputed_data.iloc[:, np.argsort(column_permutation)]
        imputed_data.columns = self.data.columns

        # Generate mask
        mask = pd.DataFrame(np.zeros_like(self.data),
                            index=self.data.index,
                            columns=self.data.columns)
        mask.iloc[grid_rows, grid_columns] = 1

        # Evaluation
        diff = np.abs(np.log(1 + self.data) - np.log(1 + imputed_data))
        mse_loss = float(np.sum(np.sum(mask * np.where(self.data != 0, 1, 0) * np.square(diff))) /
                         np.sum(np.sum(mask * np.where(self.data != 0, 1, 0))))

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
            file.write("# GOT:\n")
            grid = (imputed_data * np.where(self.data != 0, 1, np.NAN)).iloc[grid_rows, grid_columns]
            file.write(grid.to_string() + "\n")
            file.write("# GOLD STANDARD:\n")
            grid = (self.data * np.where(self.data != 0, 1, np.NAN)).iloc[grid_rows, grid_columns]
            file.write(grid.to_string() + "\n")

        log("Evaluation results saved to `%s`" % result_file)

        return metric_results
