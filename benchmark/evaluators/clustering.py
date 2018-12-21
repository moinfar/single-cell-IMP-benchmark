import os

import colorlover as cl
import numpy as np
import umap
from plotly import graph_objs as go, io as pio
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, completeness_score, calinski_harabaz_score, silhouette_score

from evaluators.base import AbstractEvaluator
from general.conf import settings
from utils.base import make_sure_dir_exists, dump_gzip_pickle, log, load_gzip_pickle
from data.data_set import get_data_set_class
from data.io import write_csv, read_table_file
from data.operations import shuffle_and_rename_columns, rearrange_and_rename_columns, normalizations, transformations
from utils.plotting import ployly_symbols


class ClusteringEvaluator(AbstractEvaluator):
    def __init__(self, uid, data_set_name=None):
        super(ClusteringEvaluator, self).__init__(uid)

        self.data_set_name = data_set_name
        self.data_set = None

    def prepare(self, **kwargs):
        if self.data_set_name is None:
            raise ValueError("data_set_name can not be None.")

        self.data_set = get_data_set_class(self.data_set_name)()
        self.data_set.prepare()

    def _load_data(self):
        if self.data_set_name == "GSE60361" or self.data_set_name == "CORTEX_3005":
            count_matrix = self.data_set.get("mRNA")
            details = self.data_set.get("details")
            classes = details.loc[["tissue",
                                   "level1class",
                                   "level2class",
                                   ]]
            return count_matrix, classes
        elif self.data_set_name.startswith("SRP041736") or self.data_set_name.startswith("POLLEN"):
            count_matrix = self.data_set.get("data")
            details = self.data_set.get("details")
            classes = details.loc[["class"]]
            return count_matrix, classes
        else:
            raise NotImplementedError()

    def generate_test_bench(self, count_file_path, **kwargs):
        preserve_columns = kwargs['preserve_columns']

        count_file_path = os.path.abspath(count_file_path)

        count_matrix, classes = self._load_data()

        # Remove zero rows
        count_matrix = count_matrix[np.sum(count_matrix, axis=1) > 0].copy()

        # Shuffle columns
        count_matrix, original_columns, column_permutation = \
            shuffle_and_rename_columns(count_matrix, disabled=preserve_columns)

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([count_matrix.to_sparse(), classes, original_columns, column_permutation],
                         hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        write_csv(count_matrix, count_file_path)
        log("Count file saved to `%s`" % count_file_path)

    def _load_hidden_state(self):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_count_matrix, classes, original_columns, column_permutation = load_gzip_pickle(hidden_data_file_path)
        count_matrix = sparse_count_matrix.to_dense()

        del sparse_count_matrix

        return count_matrix, classes, original_columns, column_permutation

    def evaluate_result(self, processed_count_file_path, result_prefix, **kwargs):
        normalization = kwargs['normalization']
        transformation = kwargs['transformation']

        # Load hidden state and data
        count_matrix, classes, original_columns, column_permutation = self._load_hidden_state()

        # Load imputed data
        imputed_data = read_table_file(processed_count_file_path)

        # Restore column names and order
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)

        # Data transformations
        imputed_data = transformations[transformation](normalizations[normalization](imputed_data))

        # Evaluation
        metric_results = dict()

        log("Fitting PCA ...")
        emb_pca = PCA(n_components=5). \
            fit_transform(imputed_data.transpose())
        log("Fitting ICA ...")
        emb_ica = FastICA(n_components=5). \
            fit_transform(imputed_data.transpose())
        log("Fitting TruncatedSVD ...")
        emb_tsvd = TruncatedSVD(n_components=5). \
            fit_transform(imputed_data.transpose())
        log("Fitting TSNE ...")
        emb_tsne_2d = TSNE(n_components=2, method='barnes_hut'). \
            fit_transform(imputed_data.transpose())
        emb_tsne = TSNE(n_components=3, method='barnes_hut'). \
            fit_transform(imputed_data.transpose())
        log("Fitting UMAP ...")
        emb_umap = umap.UMAP(n_neighbors=4, min_dist=0.3, metric='correlation'). \
            fit_transform(imputed_data.transpose())

        embedded_data = {
            "PCA": (emb_pca, emb_pca),
            "ICA": (emb_ica, emb_ica),
            "Truncated SVD": (emb_tsvd, emb_tsvd),
            "tSNE": (emb_tsne, emb_tsne_2d),
            "UMAP": (emb_umap, emb_umap)
        }

        log("Evaluating ...")
        for l in range(classes.shape[0]):
            class_names = classes.iloc[l].values
            for embedding_name in embedded_data:
                emb, emb_2d = embedded_data[embedding_name]

                k_means = KMeans(n_clusters=len(set(class_names)))
                k_means.fit(emb)
                clusters = k_means.predict(emb)

                embedding_slug = embedding_name.replace(" ", "_").lower()

                fig = go.Figure(
                    layout=go.Layout(title='%s plot' % embedding_name, font=dict(size=8)))

                if len(set(class_names)) <= 20:
                    color_scale = (cl.scales['9']['qual']['Set1'] + cl.scales['12']['qual']['Set3'])
                else:
                    color_scale = ['hsl(' + str(h) + ',50%' + ',50%)'
                                   for h in np.random.permutation(np.linspace(0, 350, len(set(class_names))))]

                for i, class_name in enumerate(list(sorted(set(class_names)))):
                    indices = [j for j, c in enumerate(class_names) if c == class_name]
                    color = color_scale[i]
                    fig.add_scatter(x=emb_2d[indices, 0], y=emb_2d[indices, 1], mode='markers',
                                    marker=dict(color=color, opacity=0.5,
                                                symbol=[ployly_symbols[c]
                                                        for c in clusters[indices]]
                                                ),
                                    name=class_name)

                pio.write_image(fig, "%s_%s_plot_%s.pdf" % (result_prefix, classes.index.values[l], embedding_slug),
                                width=800, height=600)

                metric_results.update({
                    '%s_%s_adjusted_mutual_info_score' % (embedding_slug, classes.index.values[l]):
                        adjusted_mutual_info_score(class_names, clusters, average_method="arithmetic"),
                    '%s_%s_completeness_score' % (embedding_slug, classes.index.values[l]):
                        completeness_score(class_names, clusters),
                    '%s_%s_calinski_harabaz_score' % (embedding_slug, classes.index.values[l]):
                        calinski_harabaz_score(emb, class_names),
                    '%s_%s_silhouette_score' % (embedding_slug, classes.index.values[l]):
                        silhouette_score(emb, class_names)
                })

        with open("%s_summary_all.txt" % result_prefix, 'w') as file:
            file.write("## METRICS:\n")
            for metric in sorted(metric_results):
                file.write("%s\t%4f\n" % (metric, metric_results[metric]))

            file.write("##\n## ADDITIONAL INFO:\n")

        log("Evaluation results saved to `%s_*`" % result_prefix)

        return metric_results
