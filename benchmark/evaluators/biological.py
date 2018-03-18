import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_mutual_info_score, mutual_info_score, \
    homogeneity_score, completeness_score

from evaluators.base import AbstractEvaluator
from framework.conf import settings
from framework.utils import make_sure_dir_exists, download_file_if_not_exists, \
    extract_compressed_file, get_uuid, log


class CellCyclePreservationEvaluator(AbstractEvaluator):
    DATA_SET_URL = "https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-2805/E-MTAB-2805.processed.1.zip"
    DATA_SET_MD5_SUM = "6e9f2611d670e14bb0fe750682843e10"

    def __init__(self, normalize_data_for_evaluation=True):
        self.DATA_SET_URL = CellCyclePreservationEvaluator.DATA_SET_URL
        self.DATA_SET_MD5_SUM = CellCyclePreservationEvaluator.DATA_SET_MD5_SUM

        self.DATA_SET_FILE_PATH = os.path.join("cell_cycle", os.path.basename(self.DATA_SET_URL))
        self.DATA_SET_FILE_PATH = os.path.join(settings.CACHE_DIR, self.DATA_SET_FILE_PATH)
        self.DATA_SET_DIR = "%s.d" % self.DATA_SET_FILE_PATH

        self.G1_DATA_PATH = os.path.join(self.DATA_SET_DIR, "G1_singlecells_counts.txt")
        self.G2M_DATA_PATH = os.path.join(self.DATA_SET_DIR, "G2M_singlecells_counts.txt")
        self.S_DATA_PATH = os.path.join(self.DATA_SET_DIR, "S_singlecells_counts.txt")

        self.normalize_data_for_evaluation = normalize_data_for_evaluation

    def _download_data_set(self):
        make_sure_dir_exists(os.path.dirname(self.DATA_SET_FILE_PATH))
        download_file_if_not_exists(self.DATA_SET_URL,
                                    self.DATA_SET_FILE_PATH,
                                    self.DATA_SET_MD5_SUM)

    def _extract_data_set(self):
        extract_compressed_file(self.DATA_SET_FILE_PATH, self.DATA_SET_DIR)

    def _load_and_combine_data(self):
        data_G1 = pd.read_csv(self.G1_DATA_PATH, sep="\t")
        data_G2M = pd.read_csv(self.G2M_DATA_PATH, sep="\t")
        data_S = pd.read_csv(self.S_DATA_PATH, sep="\t")

        shared_columns = ['EnsemblGeneID', 'EnsemblTranscriptID', 'AssociatedGeneName', 'GeneLength']

        merged_data = pd.merge(data_G1,
                               pd.merge(data_G2M, data_S, on=shared_columns),
                               on=shared_columns)

        merged_data[['EnsemblGeneID']] = np.where(merged_data[['AssociatedGeneName']].isna(),
                                                  merged_data[['EnsemblGeneID']],
                                                  merged_data[['AssociatedGeneName']])

        merged_data = merged_data.drop(columns=['EnsemblTranscriptID',
                                                'AssociatedGeneName',
                                                'GeneLength'])

        merged_data = merged_data.set_index('EnsemblGeneID')
        merged_data.index.names = ['Symbol']
        merged_data = merged_data.drop(['Ambiguous', 'No_feature', 'Not_aligned',
                                        'Too_low_aQual', 'Aligned'])

        assert merged_data.shape == (38385, 288)

        return merged_data

    def prepare(self):
        self._download_data_set()
        self._extract_data_set()

    def generate_test_bench(self, count_file, seed):
        np.random.seed(seed)
        uid = get_uuid()
        count_file = os.path.abspath(count_file)

        # Load dataset
        data = self._load_and_combine_data()

        # Shuffle columns
        column_permutation = np.random.permutation(data.columns)
        shuffled_data = data[column_permutation]

        # Save column list
        make_sure_dir_exists(settings.STORAGE_DIR)
        column_data_file_path = os.path.join(settings.STORAGE_DIR, "%d.json" % uid)
        with open(column_data_file_path, 'w') as json_file:
            columns = list(shuffled_data.columns)
            json.dump(columns, json_file)
        log("Benchmark hidden data saved to `%s`" % column_data_file_path)

        # rename columns and save data frame
        shuffled_data.columns = ["sample_%d" % i for i in range(len(columns))]

        make_sure_dir_exists(os.path.dirname(count_file))
        shuffled_data.to_csv(count_file, sep="\t")
        log("Count file saved to `%s`" % count_file)

        return uid

    def _laod_data_for_evaluation(self, uid, processed_count_file):
        column_data_file_path = os.path.join(settings.STORAGE_DIR, "%d.json" % uid)
        with open(column_data_file_path, 'r') as json_file:
            columns = json.load(json_file)
        imputed_data = pd.read_csv(processed_count_file, sep="\t")

        # We assume that first column is still Symbol
        imputed_data = imputed_data.set_index('Symbol')

        # Restoring original column names
        imputed_data.columns = columns

        # Remove (error correction) ERCC and mitochondrial RNAs
        remove_list = [symbol for symbol in imputed_data.index.values
                       if symbol.startswith("ERCC-") or symbol.startswith("mt-")]
        imputed_data = imputed_data.drop(remove_list)

        return imputed_data

    def evaluate_result(self, uid, processed_count_file, result_file, seed):
        data = self._laod_data_for_evaluation(uid, processed_count_file)
        gold_standard_classes = [colname.split("_")[0] for colname in data.columns.values]

        # Normalize data
        if self.normalize_data_for_evaluation:
            data = np.log10(1 + data)

        # Dimension reduction
        pca_model = PCA(n_components=10)
        low_dim_data = pca_model.fit_transform(data.transpose())

        # Simple clustering
        k_means_model = KMeans(n_clusters=3)
        clusters = k_means_model.fit_predict(low_dim_data)

        # Evaluation ...
        metric_results = {
            'adjusted_mutual_info_score': adjusted_mutual_info_score(gold_standard_classes, clusters),
            'mutual_info_score': mutual_info_score(gold_standard_classes, clusters),
            'homogeneity_score': homogeneity_score(gold_standard_classes, clusters),
            'completeness_score': completeness_score(gold_standard_classes, clusters)
        }

        results = pd.DataFrame(data={
            'Sample.Label': data.columns.values,
            'Gold.Standard.class': gold_standard_classes,
            'Clustering.Result': clusters,
        }, columns=['Sample.Label', 'Gold.Standard.class', 'Clustering.Result'])

        results = results.set_index('Sample.Label')
        results = results.sort_index()

        make_sure_dir_exists(os.path.dirname(result_file))
        with open(result_file, 'w') as file:
            file.write("## METRICS:\n")
            for metric in metric_results:
                file.write("%s: %4f\n" % (metric, metric_results[metric]))

            file.write("\n## ADDITIONAL INFO:\n")
            file.write(results.to_string() + "\n")

        log("Evaluation results saved to `%s`" % result_file)

        return metric_results
