import os

import numpy as np
import pandas as pd
import umap
from ggplot import ggplot, geom_point, ggtitle, aes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from sklearn.metrics.cluster import adjusted_mutual_info_score, completeness_score

from evaluators.base import AbstractEvaluator
from framework.conf import settings
from utils.base import make_sure_dir_exists, log, dump_gzip_pickle, load_gzip_pickle
from utils.data_set import get_data_set_class
from utils.data_table import shuffle_and_rename_columns, rearrange_and_rename_columns


class CellCyclePreservationEvaluator(AbstractEvaluator):
    def __init__(self, uid):
        super(CellCyclePreservationEvaluator, self).__init__(uid)

        self.data_set = None

    def _load_and_combine_data(self):
        data_G1 = self.data_set.get("G1")
        data_G2M = self.data_set.get("G2M")
        data_S = self.data_set.get("S")

        shared_columns = ['EnsemblGeneID', 'EnsemblTranscriptID', 'AssociatedGeneName', 'GeneLength']

        merged_data = pd.merge(data_G1,
                               pd.merge(data_G2M, data_S, on=shared_columns),
                               on=shared_columns)

        merged_data = merged_data.drop(columns=['EnsemblTranscriptID',
                                                'AssociatedGeneName',
                                                'GeneLength'])

        merged_data = merged_data.set_index('EnsemblGeneID')
        merged_data.index.names = ['Symbol']
        merged_data = merged_data.drop(['Ambiguous', 'No_feature', 'Not_aligned',
                                        'Too_low_aQual', 'Aligned'])

        assert merged_data.shape == (38385, 288)

        # remove zero-sum rows
        merged_data = merged_data[merged_data.sum(axis=1) > 0]

        return merged_data

    def prepare(self):
        data_set_class = get_data_set_class("ERP006670")
        self.data_set = data_set_class()
        self.data_set.prepare()

    def generate_test_bench(self, count_file_path, **kwargs):
        count_file_path = os.path.abspath(count_file_path)

        # Load dataset
        data = self._load_and_combine_data()

        # Shuffle columns
        new_data, original_columns, column_permutation = shuffle_and_rename_columns(data)

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([data.to_sparse(), original_columns, column_permutation], hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        new_data.to_csv(count_file_path, sep=",", index_label="")
        log("Count file saved to `%s`" % count_file_path)

        return None

    def _load_data_and_imputed_data_for_evaluation(self, processed_count_file):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_data, original_columns, column_permutation = load_gzip_pickle(hidden_data_file_path)
        data = sparse_data.to_dense()
        del sparse_data

        imputed_data = pd.read_csv(processed_count_file, sep=",", index_col=0)

        # Restoring original column names
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)

        # Remove (error correction) ERCC and mitochondrial RNAs
        remove_list = [symbol for symbol in imputed_data.index.values
                       if symbol.startswith("ERCC-") or symbol.startswith("mt-")]

        imputed_data = imputed_data.drop(remove_list)
        data = data.drop(remove_list)

        return data, imputed_data

    def evaluate_result(self, processed_count_file, result_prefix, **kwargs):
        data, imputed_data = self._load_data_and_imputed_data_for_evaluation(processed_count_file)
        gold_standard_classes = [column_name.split("_")[0] for column_name in imputed_data.columns.values]

        # Log-transform data
        if not ('no_normalize' in kwargs and kwargs['no_normalize']):
            data = np.log10(1 + data)
            imputed_data = np.log10(1 + imputed_data)

        emb_pca = PCA(n_components=10). \
            fit_transform(imputed_data.transpose())
        emb_ica = FastICA(n_components=10). \
            fit_transform(imputed_data.transpose())
        emb_tsvd = TruncatedSVD(n_components=10). \
            fit_transform(imputed_data.transpose())
        emb_tsne = TSNE(n_components=10, method='exact'). \
            fit_transform(imputed_data.transpose())
        emb_umap = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='correlation'). \
            fit_transform(imputed_data.transpose())

        # subset = np.random.choice(range(len(data.columns.values)), 30, replace=False)
        emb_lda = LinearDiscriminantAnalysis(n_components=2). \
            fit(data.transpose(), gold_standard_classes).transform(imputed_data.transpose())

        embedded_data = {
            "PCA": emb_pca,
            "ICA": emb_ica,
            "Truncated SVD": emb_tsvd,
            "tSNE": emb_tsne,
            "UMAP": emb_umap,
            "LDA": emb_lda
        }

        make_sure_dir_exists(os.path.dirname(result_prefix))
        metric_results = dict()

        for i, embedding_name in enumerate(embedded_data):
            emb = embedded_data[embedding_name]

            k_means = KMeans(n_clusters=3)
            k_means.fit(emb)
            clusters = k_means.predict(emb)

            df = pd.DataFrame(data={
                'dim_1': emb[:, 0],
                'dim_2': emb[:, 1],
                'class': gold_standard_classes,
                'cluster_symbol': [['o', 's', '^'][c] for c in clusters]
            }, columns=['dim_1', 'dim_2', 'class', 'cluster_symbol'])

            embedding_slug = embedding_name.replace(" ", "_").lower()

            p = ggplot(aes(x="dim_1", y="dim_2", color="class", shape="cluster_symbol"), data=df) + \
                geom_point(alpha=0.5, size=30) + ggtitle(embedding_name)
            p.save("%s_plot_%s.pdf" % (result_prefix, embedding_slug),
                   width=10, height=10)

            metric_results.update({
                '%s_adjusted_mutual_info_score' % embedding_slug:
                    adjusted_mutual_info_score(gold_standard_classes, clusters),
                '%s_completeness_score' % embedding_slug:
                    completeness_score(gold_standard_classes, clusters),
                '%s_calinski_harabaz_score' % embedding_slug:
                    calinski_harabaz_score(emb, gold_standard_classes),
                '%s_silhouette_score' % embedding_slug:
                    silhouette_score(emb, gold_standard_classes)
            })

        with open("%s_summary_all.txt" % result_prefix, 'w') as file:
            file.write("## METRICS:\n")
            for metric in metric_results:
                file.write("%s\t%4f\n" % (metric, metric_results[metric]))

            file.write("##\n## ADDITIONAL INFO:\n")

        log("Evaluation results saved to `%s_*`" % result_prefix)

        return metric_results
