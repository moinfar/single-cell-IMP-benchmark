from evaluators.biological import CellCyclePreservationEvaluator
from evaluators.numerical import RandomMaskedLocationPredictionEvaluator, DownSampledDataReconstructionEvaluator
from framework.conf import settings
from utils.base import generate_seed


def handle_main_arguments(args):
    settings.DEBUG = args.debug
    args.seed = int(args.seed) if args.seed else generate_seed()


def print_metric_results(results):
    for metric in results:
        print("%s: %4f" % (metric, results[metric]))


def generate_cell_cycle_test(args):
    uid = "%d_cell_cycle" % args.id
    evaluator = CellCyclePreservationEvaluator(uid)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output)


def generate_random_mask_test(args):
    uid = "%d_random_mask" % args.id
    evaluator = RandomMaskedLocationPredictionEvaluator(uid, args.data_set)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output, n_samples=args.n_samples,
                                  dropout_count=args.dropout_count)


def generate_down_sample_test(args):
    uid = "%d_down_sample" % args.id
    evaluator = DownSampledDataReconstructionEvaluator(uid, args.data_set)
    evaluator.prepare()
    evaluator.set_seed(args.seed)
    evaluator.generate_test_bench(args.output, n_samples=args.n_samples,
                                  read_ratio=args.read_ratio)


def evaluate_cell_cycle_test(args):
    uid = "%d_cell_cycle" % args.id
    evaluator = CellCyclePreservationEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_prefix,
                                        no_normalize=args.no_normalize)
    print_metric_results(results)


def evaluate_grid_mask_test(args):
    uid = "%d_random_mask" % args.id
    evaluator = RandomMaskedLocationPredictionEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_prefix,
                                        no_normalize=args.no_normalize)
    print_metric_results(results)


def evaluate_down_sample_test(args):
    uid = "%d_down_sample" % args.id
    evaluator = DownSampledDataReconstructionEvaluator(uid)
    evaluator.set_seed(args.seed)
    results = evaluator.evaluate_result(args.input, args.result_prefix,
                                        no_normalize=args.no_normalize)
    print_metric_results(results)
