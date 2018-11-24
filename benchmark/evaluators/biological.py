import os

import colorlover as cl
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import umap
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
from utils.data_table import shuffle_and_rename_columns, rearrange_and_rename_columns, read_table_file, write_csv
from utils.other import transformations, normalizations, ployly_symbols


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
        data_set_class = get_data_set_class("CELL_CYCLE")
        self.data_set = data_set_class()
        self.data_set.prepare()

    def generate_test_bench(self, count_file_path, **kwargs):
        count_file_path = os.path.abspath(count_file_path)
        rm_ercc = kwargs['rm_ercc']
        rm_mt = kwargs['rm_mt']
        rm_lq = kwargs['rm_lq']
        preserve_columns = kwargs['preserve_columns']

        # Load dataset
        data = self._load_and_combine_data()

        # Remove some rows
        if rm_ercc:
            remove_list = [symbol for symbol in data.index.values if symbol.startswith("ERCC-")]
            data = data.drop(remove_list)
        if rm_mt:
            remove_list = [symbol for symbol in data.index.values if symbol.startswith("mt-")]
            data = data.drop(remove_list)
        if rm_lq:
            remove_list = data.columns.values[data.sum(axis=0) < 1e6]
            data = data.drop(columns=remove_list)

        # Shuffle columns
        new_data, original_columns, column_permutation = shuffle_and_rename_columns(data, disabled=preserve_columns)

        # Save hidden data
        make_sure_dir_exists(settings.STORAGE_DIR)
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        dump_gzip_pickle([data.to_sparse(), original_columns, column_permutation], hidden_data_file_path)
        log("Benchmark hidden data saved to `%s`" % hidden_data_file_path)

        make_sure_dir_exists(os.path.dirname(count_file_path))
        write_csv(new_data, count_file_path)
        log("Count file saved to `%s`" % count_file_path)

        return None

    def _load_data_and_imputed_data_for_evaluation(self, processed_count_file):
        hidden_data_file_path = os.path.join(settings.STORAGE_DIR, "%s.hidden.pkl.gz" % self.uid)
        sparse_data, original_columns, column_permutation = load_gzip_pickle(hidden_data_file_path)
        data = sparse_data.to_dense()
        del sparse_data

        imputed_data = read_table_file(processed_count_file)

        # Restoring original column names
        imputed_data = rearrange_and_rename_columns(imputed_data, original_columns, column_permutation)

        # Remove (error correction) ERCC and mitochondrial RNAs
        remove_list = [symbol for symbol in imputed_data.index.values
                       if symbol.startswith("ERCC-") or symbol.startswith("mt-")]

        imputed_data = imputed_data.drop(remove_list)
        data = data.drop(remove_list)

        return data, imputed_data

    def evaluate_result(self, processed_count_file, result_prefix, **kwargs):
        normalization = kwargs['normalization']
        transformation = kwargs['transformation']

        data, imputed_data = self._load_data_and_imputed_data_for_evaluation(processed_count_file)
        gold_standard_classes = [column_name.split("_")[0] for column_name in data.columns.values]

        G1_S_related_genes = ["ENSMUSG00000000028", "ENSMUSG00000001228", "ENSMUSG00000002870", "ENSMUSG00000004642",
                              "ENSMUSG00000005410", "ENSMUSG00000006678", "ENSMUSG00000006715", "ENSMUSG00000017499",
                              "ENSMUSG00000020649", "ENSMUSG00000022360", "ENSMUSG00000022422", "ENSMUSG00000022673",
                              "ENSMUSG00000022945", "ENSMUSG00000023104", "ENSMUSG00000024151", "ENSMUSG00000024742",
                              "ENSMUSG00000025001", "ENSMUSG00000025395", "ENSMUSG00000025747", "ENSMUSG00000026355",
                              "ENSMUSG00000027242", "ENSMUSG00000027323", "ENSMUSG00000027342", "ENSMUSG00000028212",
                              "ENSMUSG00000028282", "ENSMUSG00000028560", "ENSMUSG00000028693", "ENSMUSG00000028884",
                              "ENSMUSG00000029591", "ENSMUSG00000030346", "ENSMUSG00000030528", "ENSMUSG00000030726",
                              "ENSMUSG00000030978", "ENSMUSG00000031629", "ENSMUSG00000031821", "ENSMUSG00000032397",
                              "ENSMUSG00000034329", "ENSMUSG00000037474", "ENSMUSG00000039748", "ENSMUSG00000041712",
                              "ENSMUSG00000042489", "ENSMUSG00000046179", "ENSMUSG00000055612"]
        G2_M_related_genes = ["ENSMUSG00000001403", "ENSMUSG00000004880", "ENSMUSG00000005698", "ENSMUSG00000006398",
                              "ENSMUSG00000009575", "ENSMUSG00000012443", "ENSMUSG00000015749", "ENSMUSG00000017716",
                              "ENSMUSG00000019942", "ENSMUSG00000019961", "ENSMUSG00000020330", "ENSMUSG00000020737",
                              "ENSMUSG00000020808", "ENSMUSG00000020897", "ENSMUSG00000020914", "ENSMUSG00000022385",
                              "ENSMUSG00000022391", "ENSMUSG00000023505", "ENSMUSG00000024056", "ENSMUSG00000024795",
                              "ENSMUSG00000026605", "ENSMUSG00000026622", "ENSMUSG00000026683", "ENSMUSG00000027306",
                              "ENSMUSG00000027379", "ENSMUSG00000027469", "ENSMUSG00000027496", "ENSMUSG00000027699",
                              "ENSMUSG00000028044", "ENSMUSG00000028678", "ENSMUSG00000028873", "ENSMUSG00000029177",
                              "ENSMUSG00000031004", "ENSMUSG00000032218", "ENSMUSG00000032254", "ENSMUSG00000034349",
                              "ENSMUSG00000035293", "ENSMUSG00000036752", "ENSMUSG00000036777", "ENSMUSG00000037313",
                              "ENSMUSG00000037544", "ENSMUSG00000037725", "ENSMUSG00000038252", "ENSMUSG00000038379",
                              "ENSMUSG00000040549", "ENSMUSG00000044201", "ENSMUSG00000044783", "ENSMUSG00000045328",
                              "ENSMUSG00000048327", "ENSMUSG00000048922", "ENSMUSG00000054717", "ENSMUSG00000062248",
                              "ENSMUSG00000068744", "ENSMUSG00000074802"]

        data = transformations[transformation](normalizations[normalization](data))
        imputed_data = transformations[transformation](normalizations[normalization](imputed_data))

        make_sure_dir_exists(os.path.dirname(result_prefix))

        G1_S_related_part_of_imputed_data = imputed_data.loc[G1_S_related_genes]
        G2_M_related_part_of_imputed_data = imputed_data.loc[G2_M_related_genes]

        G1_S_heatmap_fig = go.Figure(layout=go.Layout(title='G1/S Related Genes', font=dict(size=5),
                                                      xaxis=dict(title='Marker Genes', tickangle=60)))
        G2_M_heatmap_fig = go.Figure(layout=go.Layout(title='G2/M Related Genes', font=dict(size=5),
                                                      xaxis=dict(title='Marker Genes', tickangle=60)))

        def normalize(df):
            return df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)

        G1_S_heatmap_fig.add_heatmap(z=normalize(G1_S_related_part_of_imputed_data).values.T,
                                     x=G1_S_related_part_of_imputed_data.index.values,
                                     y=G1_S_related_part_of_imputed_data.columns.values,
                                     colorscale='Viridis')
        G2_M_heatmap_fig.add_heatmap(z=normalize(G2_M_related_part_of_imputed_data).values.T,
                                     x=G2_M_related_part_of_imputed_data.index.values,
                                     y=G2_M_related_part_of_imputed_data.columns.values,
                                     colorscale='Viridis')

        pio.write_image(G1_S_heatmap_fig, "%s_plot_%s.pdf" % (result_prefix, "G1_S_related_genes_heatmap"),
                        width=600, height=700)
        pio.write_image(G2_M_heatmap_fig, "%s_plot_%s.pdf" % (result_prefix, "G2_M_related_genes_heatmap"),
                        width=600, height=700)

        related_part_of_imputed_data = imputed_data.loc[G1_S_related_genes + G2_M_related_genes]
        related_part_of_original_data = data.loc[G1_S_related_genes + G2_M_related_genes]

        emb_pca = PCA(n_components=2). \
            fit_transform(related_part_of_imputed_data.transpose())
        emb_ica = FastICA(n_components=2). \
            fit_transform(related_part_of_imputed_data.transpose())
        emb_tsvd = TruncatedSVD(n_components=2). \
            fit_transform(related_part_of_imputed_data.transpose())
        emb_tsne = TSNE(n_components=2, method='exact'). \
            fit_transform(related_part_of_imputed_data.transpose())
        emb_umap = umap.UMAP(n_neighbors=4, min_dist=0.3, metric='correlation'). \
            fit_transform(related_part_of_imputed_data.transpose())

        # test best LDA classifier on original data on imputed data
        emb_lda_orig = LinearDiscriminantAnalysis(n_components=2). \
            fit(related_part_of_original_data.transpose(),
                gold_standard_classes).transform(related_part_of_imputed_data.transpose())

        # test best LDA classifier on imputed data on original data
        emb_lda_imputed = LinearDiscriminantAnalysis(n_components=2). \
            fit(related_part_of_imputed_data.transpose(),
                gold_standard_classes).transform(related_part_of_original_data.transpose())

        embedded_data = {
            "PCA": emb_pca,
            "ICA": emb_ica,
            "Truncated SVD": emb_tsvd,
            "tSNE": emb_tsne,
            "UMAP": emb_umap,
            "LDA on original data": emb_lda_orig,
            "LDA on imputed data": emb_lda_imputed
        }

        metric_results = dict()

        for i, embedding_name in enumerate(embedded_data):
            emb = embedded_data[embedding_name]

            k_means = KMeans(n_clusters=3)
            k_means.fit(emb)
            clusters = k_means.predict(emb)

            embedding_slug = embedding_name.replace(" ", "_").lower()

            fig = go.Figure(layout=go.Layout(title='%s plot using marker genes' % embedding_name, font=dict(size=8)))

            G1_indices = [i for i, c in enumerate(gold_standard_classes) if c == "G1"]
            G2M_indices = [i for i, c in enumerate(gold_standard_classes) if c == "G2M"]
            S_indices = [i for i, c in enumerate(gold_standard_classes) if c == "S"]
            fig.add_scatter(x=emb[G1_indices, 0], y=emb[G1_indices, 1], mode='markers',
                            marker=dict(color="red",
                                        symbol=[["circle-open", "diamond", "cross"][c]
                                                for c in clusters[G1_indices]]
                                        ),
                            name="G1 Phase")
            fig.add_scatter(x=emb[G2M_indices, 0], y=emb[G2M_indices, 1], mode='markers',
                            marker=dict(color="green",
                                        symbol=[["circle-open", "diamond", "cross"][c]
                                                for c in clusters[G2M_indices]]
                                        ),
                            name="G2/M Phase")
            fig.add_scatter(x=emb[S_indices, 0], y=emb[S_indices, 1], mode='markers',
                            marker=dict(color="blue",
                                        symbol=[["circle-open", "diamond", "cross"][c]
                                                for c in clusters[S_indices]]
                                        ),
                            name="S Phase")

            pio.write_image(fig, "%s_plot_%s.pdf" % (result_prefix, embedding_slug),
                            width=800, height=600)

            metric_results.update({
                '%s_adjusted_mutual_info_score' % embedding_slug:
                    adjusted_mutual_info_score(gold_standard_classes, clusters, average_method="arithmetic"),
                '%s_completeness_score' % embedding_slug:
                    completeness_score(gold_standard_classes, clusters),
                '%s_calinski_harabaz_score' % embedding_slug:
                    calinski_harabaz_score(emb, gold_standard_classes),
                '%s_silhouette_score' % embedding_slug:
                    silhouette_score(emb, gold_standard_classes)
            })

        with open("%s_summary_all.txt" % result_prefix, 'w') as file:
            file.write("## METRICS:\n")
            for metric in sorted(metric_results):
                file.write("%s\t%4f\n" % (metric, metric_results[metric]))

            file.write("##\n## ADDITIONAL INFO:\n")

        log("Evaluation results saved to `%s_*`" % result_prefix)

        return metric_results


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
                                   # "level2class",
                                   ]]
            return count_matrix, classes

    def generate_test_bench(self, count_file_path, **kwargs):
        preserve_columns = kwargs['preserve_columns']

        count_file_path = os.path.abspath(count_file_path)

        count_matrix, classes = self._load_data()

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

                for i, class_name in enumerate(list(sorted(set(class_names)))):
                    indices = [j for j, c in enumerate(class_names) if c == class_name]
                    color = (cl.scales['9']['qual']['Set1'] + cl.scales['12']['qual']['Set3'])[i]
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
