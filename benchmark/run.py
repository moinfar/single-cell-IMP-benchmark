import argparse

from main import generate_cell_cycle_test, handle_main_arguments, evaluate_grid_mask_test, \
    evaluate_cell_cycle_test, generate_random_mask_test, generate_down_sample_test, evaluate_down_sample_test


def generate_parser():
    main_parser = argparse.ArgumentParser(description="A benchmarking suite to evaluate "
                                                      "single-cell RNA-seq imputation algorithms.")

    main_parser.set_defaults(default_function=main_parser.print_help)
    main_parser.add_argument('id', metavar='ID', type=int,
                             help='unique ID to identify current benchmark.')

    # Define sub commands
    subparsers = main_parser.add_subparsers(help='action to perform')
    parser_generate = subparsers.add_parser('generate', help='generate a count file to impute')
    subparsers_generate = parser_generate.add_subparsers(help='type of benchmark')
    parser_evaluate = subparsers.add_parser('evaluate', help='evaluate a count file to impute')
    subparsers_evaluate = parser_evaluate.add_subparsers(help='type of benchmark')

    # Define global arguments
    main_parser.add_argument('--seed', '-S', metavar='N', type=int,
                             help='Seed for random generator (random if not provided)')
    main_parser.add_argument('--debug', '-D', action='store_true',
                             help='Prints debugging info')

    # Define generate commands
    parser_generate.set_defaults(default_function=parser_generate.print_help)
    parser_generate.add_argument('--output', '-o', metavar='COUNT_FILE',
                                 type=str, required=True,
                                 help='Address where noisy count matrix will be stored in')

    parser_generate_cell_cycle = subparsers_generate.add_parser('cell-cycle')
    parser_generate_cell_cycle.set_defaults(function=generate_cell_cycle_test)

    parser_generate_random_mask = subparsers_generate.add_parser('random-mask')
    parser_generate_random_mask.set_defaults(function=generate_random_mask_test)
    parser_generate_random_mask.add_argument('--data-set', '-d', metavar='DATASET-KEY',
                                             type=str, default='10xPBMC4k-GRCh38',
                                             help='Dataset to be used in this benchmark')
    parser_generate_random_mask.add_argument('--dropout-count', '-c', metavar='N',
                                             type=int, required=True,
                                             help='Number of dropouts to introduce')
    parser_generate_random_mask.add_argument('--n-samples', '-n', metavar='N',
                                             type=int, default=0,
                                             help='Number of samples (cells) used from dataset'
                                                  '(Enter 0 to use all samples.)')

    parser_generate_down_sample = subparsers_generate.add_parser('down-sample')
    parser_generate_down_sample.set_defaults(function=generate_down_sample_test)
    parser_generate_down_sample.add_argument('--data-set', '-d', metavar='DATASET-KEY',
                                             type=str, default='10xPBMC4k-GRCh38',
                                             help='Dataset to be used in this benchmark')
    parser_generate_down_sample.add_argument('--read-ratio', '-r', metavar='RATIO',
                                             type=float, required=True,
                                             help='Ratio of reads compared to original dataset')
    parser_generate_down_sample.add_argument('--n-samples', '-n', metavar='N',
                                             type=int, default=0,
                                             help='Number of samples (cells) used from dataset'
                                                  '(Enter 0 to use all samples.)')

    # Define evaluate commands
    parser_evaluate.set_defaults(default_function=parser_evaluate.print_help)
    parser_evaluate.add_argument('--input', '-i', metavar='IMPUTED_COUNT_FILE',
                                 type=str, required=True,
                                 help='Address of file containing imputed count matrix')
    parser_evaluate.add_argument('--result-prefix', '-r', metavar='RESULT_PREFIX',
                                 type=str, required=True,
                                 help='Prefix for files where evaluation result will be stored in')

    parser_evaluate_cell_cycle = subparsers_evaluate.add_parser('cell-cycle')
    parser_evaluate_cell_cycle.set_defaults(function=evaluate_cell_cycle_test)
    parser_evaluate_cell_cycle.add_argument('--no-normalize', action='store_true',
                                            help='This flag disables log-normalization step '
                                                 'in this evaluator.')

    parser_evaluate_grid_mask = subparsers_evaluate.add_parser('random-mask')
    parser_evaluate_grid_mask.set_defaults(function=evaluate_grid_mask_test)
    parser_evaluate_grid_mask.add_argument('--no-normalize', action='store_true',
                                           help='This flag disables log-normalization step '
                                                'in this evaluator.')

    parser_evaluate_down_sample = subparsers_evaluate.add_parser('down-sample')
    parser_evaluate_down_sample.set_defaults(function=evaluate_down_sample_test)
    parser_evaluate_down_sample.add_argument('--no-normalize', action='store_true',
                                             help='This flag disables log-normalization step '
                                                  'in this evaluator.')

    return main_parser


if __name__ == '__main__':
    parser = generate_parser()
    args = parser.parse_args()

    print(args)

    handle_main_arguments(args)

    if 'function' in args:
        args.function(args)
    else:
        args.default_function()
