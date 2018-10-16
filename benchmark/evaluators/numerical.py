import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from evaluators.base import AbstractEvaluator
from framework import settings
from utils.base import make_sure_dir_exists, log, dump_gzip_pickle, load_gzip_pickle
from utils.data_set import get_data_set_class
from utils.data_table import shuffle_and_rename_columns, rearrange_and_rename_columns


class RandomMaskedLocationPredictionEvaluator(AbstractEvaluator):
    def __init__(self, uid, data_set_name=None):
        super(RandomMaskedLocationPredictionEvaluator, self).__init__(uid)

        self.data_set_name = data_set_name
        self.data_set = None

    def prepare(self, **kwargs):
        if self.data_set_name is None:
            raise ValueError("data_set_name can not be None.")

        data_set_id, _ = self.data_set_name.split("-")
        self.data_set = get_data_set_class(data_set_id)()
        self.data_set.prepare()

    def _load_data(self, n_samples):
        _, data_key = self.data_set_name.split("-")
        data = self.data_set.get(data_key)
        data.columns = ["column_%d" % i for i in range(len(data.columns))]

        if n_samples is None or n_samples == 0:
            pass
        else:
            subset = np.random.choice(data.shape[1], n_samples, replace=False)
            data = data.iloc[:, subset].copy()

        return data

    def generate_test_bench(self, count_file_path, **kwargs):
        n_samples = kwargs['n_samples']
        dropout_count = kwargs['dropout_count']
        preserve_columns = kwargs['preserve_columns']

        count_file_path = os.path.abspath(count_file_path)
        data = self._load_data(n_samples)

        # Generate elimination mask
        non_zero_locations = []

        data_values = data.values
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                if data_values[x, y] > 0:
                    non_zero_locations.append((x, y))
        del data_values

        mask = np.zeros_like(data)

        masked_locations = [non_zero_locations[index] for index in
                            np.random.choice(len(non_zero_locations),
                                             dropout_count, replace=False)]

        for (x, y) in masked_locations:
            mask[x, y] = 1

        mask = pd.DataFrame(mask, index=data.index, columns=data.columns)

        # Elimination
        low_quality_data = data * (1 - mask.values)

        mask = mask[np.sum(low_quality_data, axis=1) > 0].copy()
        data = data[np.sum(low_quality_data, axis=1) > 0].copy()
        low_quality_data = low_quality_data[np.sum(low_quality_data, axis=1) > 0].copy()

        # Shuffle columns
        low_quality_data, original_columns, column_permutation = \
            shuffle_and_rename_columns(low_quality_data, disabled=preserve_columns)

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([data.to_sparse(), mask.to_sparse(), original_columns, column_permutation],
                         hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        low_quality_data.to_csv(count_file_path, sep=",", index_label="")
        log("Count file saved to `%s`" % count_file_path)

    def _load_hidden_state(self):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_data, sparse_mask, original_columns, column_permutation = load_gzip_pickle(hidden_data_file_path)
        data = sparse_data.to_dense()
        mask = sparse_mask.to_dense()

        del sparse_data
        del sparse_mask

        return data, mask, original_columns, column_permutation

    def evaluate_result(self, processed_count_file_path, result_prefix, **kwargs):
        # Load hidden state and data
        data, mask, original_columns, column_permutation = self._load_hidden_state()

        # Load imputed data
        imputed_data = pd.read_csv(processed_count_file_path, sep=",", index_col=0)

        # Restore column names and order
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)

        # Log-transform data
        if not ('no_normalize' in kwargs and kwargs['no_normalize']):
            data = np.log10(1 + data)
            imputed_data = np.log10(1 + imputed_data)

        # Evaluation
        diff = np.abs(data - imputed_data)

        mse_loss = float(np.sum(np.sum(mask * np.where(data != 0, 1, 0) * np.square(diff))) /
                         np.sum(np.sum(mask * np.where(data != 0, 1, 0))))

        metric_results = {
            'MSE': mse_loss
        }

        masked_locations = []
        mask_values = mask.values
        for x in range(mask_values.shape[0]):
            for y in range(mask_values.shape[1]):
                if mask_values[x, y] == 1:
                    masked_locations.append((x, y))

        # Save results to a file
        make_sure_dir_exists(os.path.dirname(result_prefix))
        with open("%s_summary_all.txt" % result_prefix, 'w') as file:
            file.write("## METRICS:\n")
            for metric in metric_results:
                file.write("%s\t%4f\n" % (metric, metric_results[metric]))

            file.write("##\n## ADDITIONAL INFO:\n")
            file.write("# GENE\tCELL\tGOLD_STANDARD\tRESULT:\n")
            for (x, y) in masked_locations:
                file.write("# %s\t%s\t%f\t%f\n" % (data.index.values[x],
                                                   data.columns.values[y],
                                                   data.iloc[x, y],
                                                   imputed_data.iloc[x, y]))

        log("Evaluation results saved to `%s_*`" % result_prefix)

        return metric_results


class DownSampledDataReconstructionEvaluator(AbstractEvaluator):
    def __init__(self, uid, data_set_name=None):
        super(DownSampledDataReconstructionEvaluator, self).__init__(uid)

        self.data_set_name = data_set_name
        self.data_set = None

    def prepare(self, **kwargs):
        if self.data_set_name is None:
            raise ValueError("data_set_name can not be None.")

        data_set_id, _ = self.data_set_name.split("-")
        self.data_set = get_data_set_class(data_set_id)()
        self.data_set.prepare()

    def _load_data(self, n_samples):
        _, data_key = self.data_set_name.split("-")
        data = self.data_set.get(data_key)
        data.columns = ["column_%d" % i for i in range(len(data.columns))]

        if n_samples is None or n_samples == 0:
            pass
        else:
            subset = np.random.choice(data.shape[1], n_samples, replace=False)
            data = data.iloc[:, subset].copy()

        return data

    def generate_test_bench(self, count_file_path, **kwargs):
        n_samples = kwargs['n_samples']
        read_ratio = kwargs['read_ratio']
        preserve_columns = kwargs['preserve_columns']

        count_file_path = os.path.abspath(count_file_path)
        data = self._load_data(n_samples)

        # find cumulative distribution (sum)
        data_values = data.astype(int).values
        n_all_reads = np.sum(data_values)
        data_cumsum = np.reshape(np.cumsum(data_values), data_values.shape)

        # Sample from original dataset
        new_reads = np.sort(np.random.choice(n_all_reads, int(read_ratio * n_all_reads), replace=False))

        low_quality_data = np.zeros_like(data_values)
        read_index = 0
        for x in range(data_values.shape[0]):
            for y in range(data_values.shape[1]):
                while read_index < len(new_reads) and new_reads[read_index] < data_cumsum[x, y]:
                    low_quality_data[x, y] += 1
                    read_index += 1

        # Convert to data frame
        low_quality_data = pd.DataFrame(low_quality_data, index=data.index, columns=data.columns)

        # Shuffle columns
        low_quality_data, original_columns, column_permutation = \
            shuffle_and_rename_columns(low_quality_data, disabled=preserve_columns)

        # Remove zero rows
        data = data[np.sum(low_quality_data, axis=1) > 0].copy()
        low_quality_data = low_quality_data[np.sum(low_quality_data, axis=1) > 0].copy()

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([data.to_sparse(), read_ratio, original_columns, column_permutation],
                         hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        low_quality_data.to_csv(count_file_path, sep=",", index_label="")
        log("Count file saved to `%s`" % count_file_path)

    def _load_hidden_state(self):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_data, read_ratio, original_columns, column_permutation = load_gzip_pickle(hidden_data_file_path)

        scaled_data = sparse_data.to_dense() * read_ratio

        return scaled_data, original_columns, column_permutation

    def evaluate_result(self, processed_count_file_path, result_prefix, **kwargs):
        # Load hidden state and data
        scaled_data, original_columns, column_permutation = self._load_hidden_state()

        # Load imputed data
        imputed_data = pd.read_csv(processed_count_file_path, sep=",", index_col=0)

        # Restore column names and order
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)

        # Log-transform data
        if not ('no_normalize' in kwargs and kwargs['no_normalize']):
            scaled_data = np.log10(1 + scaled_data)
            imputed_data = np.log10(1 + imputed_data)

        # Evaluation
        mse_distances = []
        euclidean_distances = []
        sqeuclidean_distances = []
        cosine_distances = []
        correlation_distances = []

        for i in range(scaled_data.shape[1]):
            x = scaled_data.values[:, i]
            y = imputed_data.values[:, i]
            mse_distances.append(np.mean(np.where(x != 0, 1, 0) * np.square(x - y)))
            euclidean_distances.append(pdist(np.vstack((x, y)), 'euclidean')[0])
            sqeuclidean_distances.append(pdist(np.vstack((x, y)), 'sqeuclidean')[0])
            cosine_distances.append(pdist(np.vstack((x, y)), 'cosine')[0])
            correlation_distances.append(pdist(np.vstack((x, y)), 'correlation')[0])

        metric_results = {
            'mean_squared_error_on_non_zeros': np.mean(mse_distances),
            'mean_euclidean_distance': np.mean(euclidean_distances),
            'mean_sqeuclidean_distance': np.mean(sqeuclidean_distances),
            'mean_cosine_distance': np.mean(cosine_distances),
            'mean_correlation_distance': np.mean(correlation_distances)
        }

        # Save results to a file
        make_sure_dir_exists(os.path.dirname(result_prefix))
        with open("%s_summary_all.txt" % result_prefix, 'w') as file:
            file.write("## METRICS:\n")
            for metric in metric_results:
                file.write("%s\t%4f\n" % (metric, metric_results[metric]))

            file.write("##\n## ADDITIONAL INFO:\n")
            file.write("# CELL\tmean_squared_error_on_non_zeros\tmean_euclidean_distance\t"
                       "mean_sqeuclidean_distance\tmean_cosine_distance\tmean_correlation_distance:\n")
            for i in range(scaled_data.shape[1]):
                file.write("# %s\t%f\t%f\t%f\t%f\t%f\n" % (scaled_data.columns.values[i],
                                                           mse_distances[i],
                                                           euclidean_distances[i],
                                                           sqeuclidean_distances[i],
                                                           cosine_distances[i],
                                                           correlation_distances[i]))

        log("Evaluation results saved to `%s_*`" % result_prefix)

        return metric_results
