master: [![pipeline status](https://gitlab.com/moinfar/deepIMP-benchmark/badges/master/pipeline.svg)](https://gitlab.com/moinfar/deepIMP-benchmark/commits/master)
dev: [![pipeline status](https://gitlab.com/moinfar/deepIMP-benchmark/badges/dev/pipeline.svg)](https://gitlab.com/moinfar/deepIMP-benchmark/commits/dev)

# deepIMP-benchmark

A benchmarking suite to evaluate single-cell RNA-seq imputation algorithms.


# Requirements

To install requirements execute:
```
pip install -r requirements.txt
```


# Usage

Go to `benchmark` directory and run command below to see the usages:
```
python run.py --help
```


# Evaluators

These ealuators are available:

- Cell-cycle preservation evaluator.
This evaluator checks whether cells in a homogeneous data set are clustered according to their cell-cycle or not.
- Random masked data prediction evaluator.
This evaluator eliminates random entries from a real data set and checks
how much the imputed values are close to eliminated nonzero values.


# Data sets

Currently, these data sets are used in evaluation:
- [ERP006670](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-2805/) containing different cell-cycle stages in mESC.
- [PBMC4k](https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k) from 10x-genomics
containing peripheral blood mononuclear cells from a healthy donor.
- [GSE60361](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361) containing 3005 high-quality single-cell profiles (with about 2M read per cell).

# Examples

#### Cell-cycle preservation evaluator.

```
python run.py 100 -S 123 generate -o counts.csv cell-cycle
```
The code above initiates a test with `100` as its id.
The test shuffles the columns of the data set `ERP006670` and saves it to `counts.csv`.
The seed for random generator is set to `123`.

The code below, evaluates an imputation algorithm assuming that the
imputation algorithm gets `counts.csv` and returns `imputed_counts.csv`.
It will cluster cells stores in `imputed_counts.csv` and check how well
the clusters fit cell-cycles. Many dimension reduction methods are performed and saved in this stage.
The result of evaluation will be printed. In addition, additional information
along results will be stored in files starting with `result_prefix`
```
python run.py 100 evaluate -i imputed_counts.csv -r result_prefix cell-cycle
```
Note that test id (`100`) should be given.

#### Random masked data prediction evaluator

```
python run.py 200 -S 123 generate -o counts.csv random-mask -n 1000 -c 20000 -d GSE60361-mm10
```
The code above initiates a test with `200` as its id.
The test generates a sample containing `1000 samples`
from `GSE60361 data set`.
It eliminates `20000 entries` from dataset.
The seed for random generator is set to `123`.

The code below, evaluates an imputation algorithm assuming that the
imputation algorithm gets `counts.csv` and returns `imputed_counts.csv`.
The result of evaluation will be printed. In addition, additional information
along results will be stored in files starting with `result_prefix`
```
python run.py 200 -S 123 evaluate -i imputed_counts.csv -r result_prefix random-mask
```
Note that test id (`200`) should be given.
