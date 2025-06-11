import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_utils import load_author_publications
from IPython.display import display
from scipy.stats import zscore

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Mapping for shortened column names
INDEX_NAME_MAP = {
    'h_index': 'h',
    'g_index': 'g',
    'i10_index': 'i10',
    'contemporary_h_index': 'hc',
    'ar_index': 'ar',
    'hpd_index': 'hpd',
    # 'trend_h_index': 'ht',
    'career_year_h_index_by_publications': 'cy_pub',
    'career_year_h_index_by_publications_year_citations': 'cy_cit',
    'career_years_h_index_by_average_citations_per_year': 'cy_avg',
    'timed_h_index_5': 'h_t5',
    'timed_h_index_10': 'h_t10',
    'ha_index': 'ha',
    'expert_index': 'expert'
}

def get_styled_index_map(index_columns):

    # Color palette (up to 10 visually distinct colors)
    color_palette = plt.get_cmap("tab10").colors

    # Marker list for clear visualization in papers/presentations
    marker_list = ["o", "s", "D", "^", "v", "*", "X", "<", ">"]

    # Cycle line styles
    linestyle_list = ["-", "--", "-."]

    index_styles = {}

    for i, col in enumerate(index_columns):
        index_styles[col] = {
            "color": color_palette[i % len(color_palette)],
            "marker": marker_list[i % len(marker_list)],
            "linestyle": linestyle_list[i % len(linestyle_list)],
        }

    return index_styles


# Apply column name shortening
def shorten_index_names(df):
    return df.rename(columns=INDEX_NAME_MAP)

def get_top_k_authors(index_df, k=20, index_cols=None):
    """
    Return Top-K author_id lists for each index.

    Parameters:
    - index_df: DataFrame containing author_id and index columns
    - k: number of top authors to return
    - index_cols: list of index column names (if None, all numeric columns are selected automatically)

    Returns:
    - top_k_dict: dictionary of {index_name: [author_id, ...]}
    """
    short_df = shorten_index_names(index_df)
    
    if index_cols is None:
        index_cols = [col for col in short_df.columns if col != "author_id" and pd.api.types.is_numeric_dtype(short_df[col])]
    
    top_k_dict = {}
    for idx in index_cols:
        top_k_dict[idx] = (
            short_df[["author_id", idx]]
            .sort_values(by=idx, ascending=False)
            .head(k)["author_id"]
            .tolist()
        )
    return top_k_dict


# Plot correlation heatmap
def plot_index_correlation_heatmap(df, method='pearson'):
    short_df = shorten_index_names(df)

    # Select only numeric columns for analysis, excluding identifiers
    exclude_cols = {'author_id', 'recency_score', 'quality_score', 'paper_count'}
    index_cols = [col for col in short_df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(short_df[col])]

    # Compute and visualize correlation
    corr = short_df[index_cols].corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(f"{method.capitalize()} Correlation between Indices")
    plt.tight_layout()
    plt.show()
    
    return corr

def print_top_k_authors(df, k=10, index_columns=None, id_col="author_id"):
    print(f"Top-{k} Authors by Each Index:\n")
    for col in index_columns:
        print(f"{col}")
        top_authors = df[[id_col, col]].sort_values(by=col, ascending=False).head(k)
        print(top_authors.to_string(index=False))
        print("-" * 40)

# Compute overlap counts between Top-K authors
def compute_top_k_overlap_matrix(df, index_columns, k=10, id_col="author_id"):
    overlap_matrix = pd.DataFrame(index=index_columns, columns=index_columns)

    # Build Top-K author sets for each index
    top_k_dict = get_top_k_authors(df, k, index_columns)
    top_k_sets = {col: set(ids) for col, ids in top_k_dict.items()}

    # Cross compare to count overlaps
    for col1 in index_columns:
        for col2 in index_columns:
            overlap = len(top_k_sets[col1].intersection(top_k_sets[col2]))
            ratio = overlap / k
            overlap_matrix.loc[col1, col2] = ratio

    return overlap_matrix.round(1)

# Plot heatmap
def plot_top_k_overlap_heatmap(overlap_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(overlap_matrix, annot=True, fmt="d", cmap="YlGnBu", square=True)
    plt.title("Top-K Author Overlap Between Indices")
    plt.tight_layout()
    plt.show()

def find_index_specific_top_k_authors(df, index_columns, k=10, id_col="author_id"):
    top_k_dict = get_top_k_authors(df, k, index_columns)
    top_k_sets = {col: set(ids) for col, ids in top_k_dict.items()}

    results = {}
    for col in index_columns:
        others = set().union(*[v for k2, v in top_k_sets.items() if k2 != col])
        unique_authors = top_k_sets[col] - others
        results[col] = unique_authors
    return results

def plot_rank_shift(df, index_columns, base_index, top_n=15, id_col="author_id"):
    rank_df = df[[id_col] + index_columns].copy()
    for col in index_columns:
        rank_df[f"{col}_rank"] = rank_df[col].rank(ascending=False, method='min')

    base_rank = rank_df[f"{base_index}_rank"]
    shift_data = {}
    for col in index_columns:
        if col == base_index:
            continue
        shift = (rank_df[f"{col}_rank"] - base_rank).abs()
        top_authors = shift.sort_values(ascending=False).head(top_n).index
        shift_data[col] = shift.loc[top_authors]

    shift_df = pd.DataFrame(shift_data)
    shift_df.plot(kind='bar', figsize=(12, 6), title=f"Top-{top_n} Rank Shift Compared to '{base_index}'")
    plt.ylabel("Absolute Rank Difference")
    plt.xlabel("Author Index")
    plt.tight_layout()
    plt.show()

def plot_unique_author_counts_by_k(df, index_columns, k_values, id_col="author_id"):
    data = []

    # Compute the number of unique Top-K authors
    for k in k_values:
        top_k_dict = get_top_k_authors(df, k, index_columns)
        top_k_sets = {col: set(ids) for col, ids in top_k_dict.items()}

        for col in index_columns:
            others = set().union(*[v for k2, v in top_k_sets.items() if k2 != col])
            unique_authors = top_k_sets[col] - others
            data.append({
                "index": col,
                "k": k,
                "unique_authors": len(unique_authors)
            })

    result_df = pd.DataFrame(data)

    # Retrieve style map
    index_styles = get_styled_index_map(index_columns)

    # Visualization
    plt.figure(figsize=(12, 6))
    for col in index_columns:
        subset = result_df[result_df["index"] == col]
        style = index_styles[col]
        plt.plot(
            subset["k"],
            subset["unique_authors"],
            label=col,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"]
        )

    # plt.title("Top-K Only Authors per Index")
    plt.ylabel("Number of Unique Researchers")
    plt.xlabel("k")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("unique_author_counts.png", dpi=600)
    plt.show()
    
    return result_df


def plot_index_clustermap(overlap_matrix, k):
    # Convert similarity matrix to distance matrix
    dist_matrix = 1 - overlap_matrix.astype(float) / k

    # Visualize clustermap
    sns.clustermap(dist_matrix, cmap="Blues", annot=True, fmt=".2f")
    plt.title("Index Clustering by Top-K Overlap")
    plt.show()

# Example visualization: ha-index vs h-index
def plot_index_scatter(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def analyze_unique_authors_characteristics(unique_authors, data_dir, current_year=2013):
    all_years = []
    all_citations = []
    paper_counts = []
    
    for author_id in unique_authors:
        df = load_author_publications(data_dir, author_id, current_year)
        if df.empty:
            continue
        all_years.extend(df["year"])
        all_citations.extend(df["citations"])
        paper_counts.append(len(df))

    # 1. Plot distribution of publication years
    plt.figure(figsize=(10, 4))
    sns.histplot(all_years, bins=range(min(all_years), current_year+1), kde=False)
    plt.title("Publication Year Distribution of Unique Authors")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.tight_layout()
    plt.show()

    # 2. Plot citation distribution
    plt.figure(figsize=(10, 4))
    sns.histplot(all_citations, bins=20, kde=False)
    plt.title("Citation Distribution of Unique Authors")
    plt.xlabel("Citations per Paper")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 3. Print average paper and citation counts
    print(f"Number of authors: {len(paper_counts)}")
    print(f"Average paper count: {sum(paper_counts)/len(paper_counts):.2f}")
    print(f"Average citations: {sum(all_citations)/len(all_citations):.2f}")

def analyze_extreme_rank_shift(df, index_columns, base_index='expert', top_n=5, id_col='author_id'):
    # Calculate rankings for all indices
    rank_df = df[[id_col] + index_columns].copy()
    for col in index_columns:
        rank_df[f"{col}_rank"] = rank_df[col].rank(ascending=False, method='min')

    base_rank = rank_df[f"{base_index}_rank"]

    # Compute rank differences relative to the base index
    for col in index_columns:
        if col != base_index:
            rank_df[f"{col}_rank_diff"] = (rank_df[f"{col}_rank"] - base_rank).abs()

    # Select authors with the largest change
    rank_df["max_rank_diff"] = rank_df[[f"{col}_rank_diff" for col in index_columns if col != base_index]].max(axis=1)
    top_authors = rank_df.sort_values(by="max_rank_diff", ascending=False).head(top_n)

    # Display result table
    display_cols = [id_col] + index_columns + [f"{col}_rank" for col in index_columns]
    print(f"Top-{top_n} authors with large rank changes relative to {base_index}")
    
    display(top_authors[display_cols])
    
    return top_authors[display_cols]

def plot_author_index_and_activity_time_series(author_id, index_df, data_dir, selected_columns=None, normalize_zscore=True):
    """
    Visualize index time series and paper/citation counts for a specific author.
    """
    short_df = shorten_index_names(index_df)

    if selected_columns is None:
        exclude_cols = {"author_id", "year", "quality_score", "recency_score"}
        selected_columns = [
            col for col in short_df.columns
            if col not in exclude_cols and short_df[col].dtype != "object"
        ]

    styled_map = get_styled_index_map(index_columns=selected_columns)
    plot_df = short_df.copy()

    if normalize_zscore:
        plot_df[selected_columns] = plot_df[selected_columns].apply(zscore, nan_policy='omit')

    # Index time series plot
    plt.figure(figsize=(12, 6))
    for col in selected_columns:
        if col in plot_df.columns:
            style = styled_map.get(col, {})
            plt.plot(
                plot_df["year"], plot_df[col],
                label=col,
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                color=style.get("color", "black")
            )

    plt.title(f"Index Time Series for {author_id} (Z-Score Normalized)")
    plt.xlabel("Year")
    plt.ylabel("Z-Score" if normalize_zscore else "Raw Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Paper and citation visualization
    df_papers = load_author_publications(data_dir, author_id, current_year=2013)
    df_papers["year"] = df_papers["year"].astype(int)
    
    paper_stats = df_papers.groupby("year")["citations"].agg(["count", "sum"]).reset_index()
    paper_stats.rename(columns={"count": "papers", "sum": "citations"}, inplace=True)
    
    full_years = pd.DataFrame({"year": range(1970, 2013 + 1)})
    paper_stats = full_years.merge(paper_stats, on="year", how="left").fillna(0)
    
    paper_stats["papers"] = paper_stats["papers"].astype(int)
    paper_stats["citations"] = paper_stats["citations"].astype(int)

    if not paper_stats.empty:
        paper_stats.set_index("year")[["papers"]].plot(
            kind="bar", figsize=(12, 4), title=f"Number of Papers per Year: {author_id}", color="skyblue"
        )
        plt.ylabel("Papers")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()

        paper_stats.set_index("year")[["citations"]].plot(
            kind="bar", figsize=(12, 4), title=f"Number of Citations per Year: {author_id}", color="salmon"
        )
        plt.ylabel("Citations")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()
        
        return plot_df, paper_stats
    else:
        print("No paper data available, skipping activity visualization.")
        return plot_df, paper_stats

