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

Run `python run.py --help` to see the usage.


# Evaluators

These ealuators are available:

- Cell-cycle preservation evaluator.
This evaluator checks whether cells in a homogeneous data set are clustered according to their cell-cycle or not.
- Grid masked data prediction evaluator. This evaluator eliminates a random grid from a real data set and checks how much the imputed values are close to eliminated nonzero values.


# Data sets

Currently, these data sets are used in evaluation:
- [ERP006670](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-2805/) containing different cell-cycle stages in mESC.
- [PBMC4k](https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k) from 10x-genomics
containing peripheral blood mononuclear cells from a healthy donor.
- [GSE60361](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361) containing 3005 high-quality single-cell profiles (with about 2M read per cell).

# Examples

#### Cell-cycle preservation evaluator.

```
python run.py generate cell-cycle 100 -o counts.txt -S 123
```
The code above initiates a test with `100` as its id.
The test shuffles the columns of the data set `ERP006670` and
saves it to `counts.txt`.
The seed for random generator is set to `123`.

The code below, evaluates an imputation algorithm assuming that the
imputation algorithm gets `counts.txt` and returns `imputed_counts.txt`.
It will cluster cells stores in `imputed_counts.txt` and check how well
the clusters fit cell-cycles.
The result of evaluation will be printed. In addition, additional information
along results will be stored in `result.txt`
```
python run.py evaluate cell-cycle 100 -i counts.txt -r result.txt
```

#### Grid masked data prediction evaluator

```
python run.py generate grid-prediction 200 -o counts.txt -d 10xPBMC4k-GRCh38 -n 500 -g 0.2x0.3 -S 123
```
The code above initiates a test with `200` as its id.
The test generates a sample containing `500 samples`
from `10xPBMC4k data set`.
It eliminates a box with size (`0.2 of genes x 0.3 of cells`).
The seed for random generator is set to `123`.

The code below, evaluates an imputation algorithm assuming that the
imputation algorithm gets `counts.txt` and returns `imputed_counts.txt`.
The result of evaluation will be printed. In addition, additional information
along results will be stored in `result.txt`
```
python run.py evaluate grid-prediction 200 -i imputed_counts.txt -r result.txt
```
