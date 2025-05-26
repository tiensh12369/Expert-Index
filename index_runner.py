import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from index_evaluator import load_authors_data, evaluate_all_with_expert_index
from data_utils import load_author_publications, load_author_index_time_series
from index_metrics import (
    h_index, g_index, i10_index, ha_index, ar_index,
    timed_h_index, contemporary_h_index, trend_h_index,
    career_years_h_index_by_average_citations_per_year,
)
from viz_utils import *




authors_file = "./gsc_data/authors.all"
data_dir = "./gsc_data/DATA/"
output_file = "all_index_results.csv"
current_year = 2013

print("Calculating or loading exponents...")
authors_data = load_authors_data(authors_file)
if not os.path.exists(output_file):
    index_df, max_q, max_r = evaluate_all_with_expert_index(authors_data, data_dir, current_year)
    print(max_q, max_r)
    index_df.to_csv(output_file, index=False)
else:
    index_df = pd.read_csv(output_file)
    
start_year = 1970
end_year = 2013

authors_data = load_authors_data(authors_file)

expert_by_year_file = f"index_by_{end_year}.csv"
if not os.path.exists(expert_by_year_file):
    print(f"Calculating the full author year-by-year index...{expert_by_year_file}")
    index_df, max_q, max_r = evaluate_all_with_expert_index(authors_data, data_dir, end_year)
    print(max_q, max_r)
    index_df.to_csv(expert_by_year_file, index=False)
    print(f"Completion of calculation for {end_year}")
else:
    index_df = pd.read_csv(expert_by_year_file)
    print(f"Load completed in {end_year}")
    
index_df_short = index_df.rename(columns=INDEX_NAME_MAP)
index_columns = list(INDEX_NAME_MAP.values())

corr_pearson = plot_index_correlation_heatmap(index_df, method='pearson')
mask = np.triu(np.ones_like(corr_pearson, dtype=bool), k=1)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_pearson, annot=True, annot_kws={"size": 10}, cmap="coolwarm", mask=mask)
plt.tight_layout()
plt.savefig("pearson_heatmap.png", dpi=600)

corr_spearman = plot_index_correlation_heatmap(index_df, method='spearman')

mask = np.triu(np.ones_like(corr_spearman, dtype=bool), k=1)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_spearman, annot=True, annot_kws={"size": 10}, cmap="coolwarm", mask=mask)
plt.tight_layout()
plt.savefig("spearman_heatmap.png", dpi=600)

print("Comparing and Overlapping Top-K...")
overlap_matrix = compute_top_k_overlap_matrix(index_df_short, index_columns, k=800)
mask = np.triu(np.ones_like(overlap_matrix, dtype=float), k=1)
plt.figure(figsize=(12, 10))
sns.heatmap(overlap_matrix.astype(float), annot=True, fmt=".2f", annot_kws={"size": 10}, cmap="YlGnBu", mask=mask)
plt.tight_layout()
plt.savefig("top_k_overlap_heatmap.png", dpi=600)

print("Analyzing independent authors...")
authors_count = 0
unique_authors_by_index = find_index_specific_top_k_authors(index_df_short, index_columns, k=5)
for idx, authors in unique_authors_by_index.items():
    print(f"{len(authors)} authors included in the {idx} index as Top-K only: {authors}")
    authors_count += len(authors)

print("Visualizing the number of unique authors according to Top-K expansion...")
k_range = list(range(10, 110, 10))
author_counts_by_k = plot_unique_author_counts_by_k(index_df_short, index_columns, k_values=k_range)

author_id = "sUVeH-4AAAAJ"
index_df_single = load_author_index_time_series(author_id, "./", 2009, 2013)
plot_df, paper_stats = plot_author_index_and_activity_time_series(author_id, index_df_single, data_dir)

current_year = 2013

summary_stats = []
temporal_activity = defaultdict(list)

top_k_dict = get_top_k_authors(index_df, k=800, index_cols=None)

for index_name, author_ids in top_k_dict.items():
    for author_id in author_ids:
        df = load_author_publications(data_dir, author_id, current_year)
        df = df[(df["year"] >= 1970) & (df["year"] <= current_year)]  # ✅ 연도 필터링
        if df.empty:
            continue

        first_year = df["year"].min()
        last_year = df["year"].max()
        career_years = last_year - first_year + 1

        summary_stats.append({
            "index": index_name,
            "author_id": author_id,
            "papers": len(df),
            "total_citations": df["citations"].sum(),
            "avg_citations": df["citations"].mean(),
            "recent_5y_citations": df[df["year"] >= current_year - 5]["citations"].sum(),
            "career_years": career_years,
            "avg_year": df["year"].mean()
        })

        year_group = df.groupby("year")["citations"].sum()
        for year, c in year_group.items():
            temporal_activity[(index_name, year)].append(c)

summary_df = pd.DataFrame(summary_stats)

print("\n [Summary of Basic Statistics for Independent Authors]")
summary_table = summary_df.groupby("index").agg({
    "papers": ["mean", "std"],
    "total_citations": ["mean", "std"],
    "avg_citations": ["mean", "std"],
    "recent_5y_citations": ["mean", "std"],
    "career_years": ["mean", "std"],
    "avg_year": ["mean", "std"]
}).round(2)

weighted_avg_citations = (
    summary_df.groupby("index").apply(lambda df: df["total_citations"].sum() / df["papers"].sum())
).round(2)

summary_table[("avg_citations", "weighted")] = weighted_avg_citations

display(summary_table)

summary_table.to_excel("independent_author_stats.xlsx")
print("Save complete: independent_author_stats.xlsx")

def show_author_summary(author_id, index_df, data_dir, current_year=2013):
    if author_id not in index_df["author_id"].values:
        print(f"The corresponding author_id ({author_id}) does not exist in index_df.")
        return

    row = index_df[index_df["author_id"] == author_id].squeeze()
    rank_df = index_df.copy()
    for col in index_df.columns:
        if col not in {"author_id", "quality_score", "recency_score"} and pd.api.types.is_numeric_dtype(index_df[col]):
            rank_df[f"{col}_rank"] = index_df[col].rank(method="min", ascending=False)
    row_rank = rank_df[rank_df["author_id"] == author_id].squeeze()

    df = load_author_publications(data_dir, author_id, current_year=current_year)
    df = df[(df["year"] >= 2009) & (df["year"] <= current_year)]
    if df.empty or "year" not in df.columns:
        print("The paper data does not exist.")
        return

    first_year = int(df["year"].min())
    last_year = int(df["year"].max())
    career_years = last_year - first_year + 1
    paper_count = len(df)
    total_citations = int(df["citations"].sum())

    metrics = [
        col for col in index_df.columns 
        if col not in {"author_id", "quality_score", "recency_score"}
        and pd.api.types.is_numeric_dtype(index_df[col])
    ]

    data = {
        "author_id": author_id,
        "paper_count": paper_count,
        "citation_count": total_citations,
        "first_pub_year": first_year,
        "last_pub_year": last_year,
        "career_years": career_years
    }

    for col in metrics:
        data[col] = row[col]
        data[f"{col}_rank"] = int(row_rank[f"{col}_rank"])

    return pd.DataFrame([data])

top_k_dict = get_top_k_authors(index_df, k=5, index_cols=None)
all_unique_authors = sorted(set().union(*top_k_dict.values()))
summary_dfs = []

for author_id in all_unique_authors:
    df = show_author_summary(author_id, index_df, data_dir)
    if df is not None:
        summary_dfs.append(df)

final_summary = pd.concat(summary_dfs, ignore_index=True)
display(final_summary.head())
output_path = "unique_authors_summary_top5_2009.csv"
final_summary.to_csv(output_path, index=False, encoding="utf-8-sig")
print(len(final_summary))
print(f"Save complete: {output_path}")


