# Expert Index

## Overview
The Expert Index project provides tools to evaluate academic authors using a variety of citation-based metrics and a combined "expert" score. It loads publication records from Google Scholar style exports and produces rankings and visualizations that highlight how different metrics correlate.

## Dependencies and Setup
This repository requires **Python 3.8** or newer and the following libraries:

- pandas
- numpy
- seaborn
- matplotlib

Install the dependencies with `pip install pandas numpy seaborn matplotlib` or by using your preferred environment manager.

## Data Directory Structure
Place your raw data under a directory named `gsc_data` in the project root:

```
gsc_data/
├── authors.all                # metadata for all authors
└── DATA/
    ├── <AUTHOR_ID>_.dat       # publication data for each author
    └── ...
```

The `.dat` files contain `citations|year` pairs. The `authors.all` file is expected to have six pipe-separated fields for each author.

## Running the Code
The easiest way to generate results is by executing `index_runner.py`:

```bash
python index_runner.py
```

The same logic is available in the notebooks `index_runner.ipynb`, `index_runner_2013.ipynb`, and `index_runner_all.ipynb` for interactive exploration. These notebooks assume the data directory structure described above.

## Outputs
Running the script produces CSV files with computed scores (for example `all_index_results.csv` and `index_by_<year>.csv`) as well as several heatmap images:

- `pearson_heatmap.png` and `spearman_heatmap.png` – correlation matrices between metrics
- `top_k_overlap_heatmap.png` – overlap between the top authors for each metric
- `unique_author_counts.png` – how many authors are uniquely ranked in each metric

A summary CSV named `unique_authors_summary_top5_2009.csv` is also generated for the top authors.
