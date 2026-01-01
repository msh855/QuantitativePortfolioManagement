#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 18:08:04 2022

@author: safishajjouz
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats_lumi as qs
import numpy as np
from typing import Union


def scatter_plot_simple(df, x: str, y: str):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")

    # expects as index the labels

    plt.figure(figsize=[15, 7])
    sns.regplot(data=df, x=x, y=y, fit_reg=False, marker="o", color="skyblue", scatter_kws={"s": 400})

    # add annotations one by one with a loop
    for line in range(0, df.shape[0]):
        plt.text(
            df[x][line],
            df[y][line],
            df.index[line],
            horizontalalignment="left",
            size="medium",
            color="black",
            weight="semibold",
        )

    plt.show()


def correlation_matrix(
    df: pd.DataFrame,
    corr_limit: float = None,
    phi_correlation: bool = False,
    figsize: tuple = (18, 10),
    diagonal=True,
    **kwargs
):
    # check if df is a dataframe
    if not isinstance(df, pd.DataFrame):
        raise ValueError("you must pass a Pandas DataFrame")
    df_corr = df.corr()

    if phi_correlation:
        df_corr = df.phik_matrix()

    df_corr = round(df_corr, 2)

    if corr_limit:
        df_corr = df_corr[df_corr < corr_limit]

    # Generate a mask for the upper triangle
    if diagonal:
        mask = np.triu(np.ones_like(df_corr, dtype=bool))
    else:
        mask = None

    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap="BrBG", **kwargs)
    heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 18}, pad=12)


def monthly_heatmap(
    returns,
    annot_size=10,
    figsize=(10, 5),
    cbar=True,
    square=False,
    compounded=True,
    eoy=True,
    grayscale=False,
    fontname="Arial",
    ylabel=True,
    savefig=None,
    show=True,
):
    # colors, ls, alpha = _core._get_colors(grayscale)
    cmap = "gray" if grayscale else "RdYlGn"

    returns = qs.stats.monthly_returns(returns, eoy=eoy, compounded=compounded) * 100

    fig_height = len(returns) / 3

    if figsize is None:
        size = list(plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0] * 1.04, max([fig_height, figsize[1]]))

    fig, ax = plt.subplots(figsize=figsize)

    # plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_title(
        "Monthly Returns (%)\n", fontsize=14, y=0.995, fontname=fontname, fontweight="bold", color="black", pad=20
    )

    # _sns.set(font_scale=.9)

    ax = sns.heatmap(
        returns,
        ax=ax,
        annot=True,
        center=0,
        annot_kws={"size": annot_size},
        fmt="0.2f",
        linewidths=0.5,
        square=square,
        cbar=cbar,
        cmap=cmap,
        cbar_kws={"format": "%.0f%%"},
    )
    # _sns.set(font_scale=1)

    # align plot to match other
    if ylabel:
        ax.set_ylabel("Years", fontname=fontname, fontweight="bold", fontsize=12)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.tick_params(colors="#808080")
    plt.xticks(rotation=0, fontsize=annot_size * 1.2)
    plt.yticks(rotation=0, fontsize=annot_size * 1.2)

    try:
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    if show:
        plt.show(block=False)

    plt.close()

    if not show:
        return fig

    return


# fig = px.scatter(df_perf_python, x="AnnualizedStandardDeviation", y="AnnualizedReturn",
#                 color = 'AnnualizedSharpe', size ='AnnualizedSharpe', template='plotly_dark',
#                   hover_data=['Company', 'ticker'],
#                   color_continuous_scale=px.colors.sequential.Viridis, title = 'S&P 500 Companies Perfomance the Last 4 months')
# fig.show()
# fig.write_html("s_p_perf.html")


# rp.plot_dendrogram(returns=ret_log, codependence='pearson',
#                        linkage='ward', max_k=20, bins_info = 3,
#                        leaf_order=True, ax=None)


### Quick Check on Correlations (linear and Tails)
# df_corr = df_rolling.corr()
# df_corr = df_corr[df_corr<0.4]
# plt.figure(figsize=(18, 10))
# heatmap = sns.heatmap(df_corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

### Hierarchical Clustering
# rp.plot_clusters(returns=ret, codependence="spearman",
#                     linkage='ward', max_k=10,
#                    leaf_order=True, dendrogram=True, ax=None)

# rp.plot_clusters(returns=df_rolling, codependence="pearson",
#                       linkage='ward', max_k=20,
#                       leaf_order=True, dendrogram=True, ax=None)

# rp.plot_dendrogram(returns=ret_simple, codependence='pearson',
#                         linkage='complete', max_k=20,
#                         leaf_order=True, ax=None)

# rp.plot_network(returns=ret, codependence="spearman",
#                      linkage="ward", k=None, max_k=10,
#                      alpha_tail=0.05, leaf_order=True,
#                      kind='spring', ax=None)

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
#
# fig = px.line(df.reset_index(), x='Date', y=df.columns, width=1000, height=800,
#               title='Time Series with Range Slider and Selectors')
#
# fig.update_xaxes(
#     rangeslider_visible=True,
#     rangeselector=dict(
#         buttons=list([
#             dict(count=1, label="1m", step="month", stepmode="backward"),
#             dict(count=6, label="6m", step="month", stepmode="backward"),
#             dict(count=1, label="YTD", step="year", stepmode="todate"),
#             dict(count=1, label="1y", step="year", stepmode="backward"),
#             dict(step="all")
#         ])
#     )
# )
# fig.show()


def plot_bootstrap_distribution(bootstrap_results: Union[pd.DataFrame, pd.Series],
                                 metric: str = None,
                                 confidence_level: float = 0.95,
                                 figsize: tuple = (10, 6),
                                 color: str = 'steelblue',
                                 title: str = None,
                                 savefig: str = None,
                                 show: bool = True):
    """
    Plot the bootstrap distribution for a single metric with confidence intervals.
    
    Args:
        bootstrap_results: DataFrame with bootstrap samples (columns are metrics) or Series for single metric
        metric: Name of the metric column to plot (required if bootstrap_results is DataFrame)
        confidence_level: Confidence level for intervals (default: 0.95 for 95% CI)
        figsize: Figure size as (width, height)
        color: Color for the histogram
        title: Custom title for the plot
        savefig: Path to save the figure (if None, figure is not saved)
        show: Whether to display the plot
    
    Returns:
        matplotlib figure object if show=False, otherwise None
    """
    # Handle input
    if isinstance(bootstrap_results, pd.DataFrame):
        if metric is None:
            raise ValueError("metric parameter is required when bootstrap_results is a DataFrame")
        if metric not in bootstrap_results.columns:
            raise ValueError(f"Metric '{metric}' not found in bootstrap_results columns")
        data = bootstrap_results[metric].dropna()
    elif isinstance(bootstrap_results, pd.Series):
        data = bootstrap_results.dropna()
        metric = bootstrap_results.name if bootstrap_results.name else 'metric'
    else:
        raise ValueError("bootstrap_results must be a pandas DataFrame or Series")
    
    # Calculate statistics
    mean_val = data.mean()
    median_val = data.median()
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    ci_lower = np.percentile(data, lower_percentile)
    ci_upper = np.percentile(data, upper_percentile)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    n, bins, patches = ax.hist(data, bins=50, density=True, alpha=0.7, 
                                color=color, edgecolor='black', linewidth=0.5)
    
    # Add KDE
    from scipy import stats as scipy_stats
    kde = scipy_stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Add vertical lines for statistics
    ax.axvline(mean_val, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    ax.axvline(ci_lower, color='red', linestyle=':', linewidth=2, 
               label=f'{int(confidence_level*100)}% CI Lower: {ci_lower:.4f}')
    ax.axvline(ci_upper, color='red', linestyle=':', linewidth=2, 
               label=f'{int(confidence_level*100)}% CI Upper: {ci_upper:.4f}')
    
    # Labels and title
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    
    if title is None:
        title = f'Bootstrap Distribution: {metric.replace("_", " ").title()}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    # Show or return
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig


def plot_bootstrap_distributions(bootstrap_results: pd.DataFrame,
                                  metrics: list = None,
                                  confidence_level: float = 0.95,
                                  plot_type: str = 'violin',
                                  figsize: tuple = None,
                                  title: str = None,
                                  savefig: str = None,
                                  show: bool = True):
    """
    Plot multiple bootstrap distributions for comparison.
    
    Args:
        bootstrap_results: DataFrame with bootstrap samples (columns are metrics)
        metrics: List of metric names to plot (if None, plots all columns)
        confidence_level: Confidence level for intervals (default: 0.95)
        plot_type: Type of plot - 'violin', 'box', or 'hist' (default: 'violin')
        figsize: Figure size as (width, height). If None, auto-calculated
        title: Custom title for the plot
        savefig: Path to save the figure (if None, figure is not saved)
        show: Whether to display the plot
    
    Returns:
        matplotlib figure object if show=False, otherwise None
    """
    if not isinstance(bootstrap_results, pd.DataFrame):
        raise ValueError("bootstrap_results must be a pandas DataFrame")
    
    # Select metrics
    if metrics is None:
        metrics = bootstrap_results.columns.tolist()
    else:
        missing = [m for m in metrics if m not in bootstrap_results.columns]
        if missing:
            raise ValueError(f"Metrics not found in bootstrap_results: {missing}")
    
    data = bootstrap_results[metrics].copy()
    n_metrics = len(metrics)
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        if plot_type == 'hist':
            cols = min(3, n_metrics)
            rows = int(np.ceil(n_metrics / cols))
            figsize = (6 * cols, 5 * rows)
        else:
            figsize = (max(10, n_metrics * 1.5), 6)
    
    if plot_type == 'hist':
        # Create subplots for histograms
        cols = min(3, n_metrics)
        rows = int(np.ceil(n_metrics / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Ensure axes is always a list
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            metric_data = data[metric].dropna()
            
            # Calculate statistics
            mean_val = metric_data.mean()
            alpha = 1 - confidence_level
            ci_lower = np.percentile(metric_data, (alpha / 2) * 100)
            ci_upper = np.percentile(metric_data, (1 - alpha / 2) * 100)
            
            # Plot histogram
            ax.hist(metric_data, bins=30, density=True, alpha=0.7, 
                   color='steelblue', edgecolor='black', linewidth=0.5)
            
            # Add mean and CI
            ax.axvline(mean_val, color='darkgreen', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(ci_lower, color='red', linestyle=':', linewidth=1.5, label=f'{int(confidence_level*100)}% CI')
            ax.axvline(ci_upper, color='red', linestyle=':', linewidth=1.5)
            
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        if title is None:
            title = 'Bootstrap Distributions for All Metrics'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        
    elif plot_type == 'violin':
        # Create violin plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for violin plot
        data_melted = data.melt(var_name='Metric', value_name='Value')
        
        # Create violin plot
        sns.violinplot(data=data_melted, x='Metric', y='Value', ax=ax, 
                      palette='Set2', inner='box')
        
        # Add mean points
        means = data.mean()
        ax.scatter(range(len(means)), means, color='red', s=100, zorder=3, 
                  marker='D', label='Mean', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], 
                          rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        if title is None:
            title = 'Bootstrap Distributions Comparison (Violin Plot)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
    elif plot_type == 'box':
        # Create box plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        bp = ax.boxplot([data[m].dropna() for m in metrics], 
                        labels=[m.replace('_', ' ').title() for m in metrics],
                        patch_artist=True, notch=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        if title is None:
            title = 'Bootstrap Distributions Comparison (Box Plot)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        raise ValueError("plot_type must be 'violin', 'box', or 'hist'")
    
    plt.tight_layout()
    
    # Save if requested
    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    # Show or return
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig


def plot_bootstrap_comparison(bootstrap_results: pd.DataFrame,
                              comparison_col: str = None,
                              metrics: list = None,
                              confidence_level: float = 0.95,
                              figsize: tuple = None,
                              title: str = None,
                              savefig: str = None,
                              show: bool = True):
    """
    Plot comparison of bootstrap distributions for in-sample vs out-of-sample or other comparisons.
    
    This function expects a DataFrame where column names include suffixes like '_in_sample', 
    '_out_sample', or similar identifiers for comparison.
    
    Args:
        bootstrap_results: DataFrame with bootstrap samples where columns contain comparison identifiers
        comparison_col: Substring to identify comparison groups (e.g., 'sample' for in_sample/out_sample)
        metrics: List of base metric names to compare (if None, auto-detects from columns)
        confidence_level: Confidence level for intervals (default: 0.95)
        figsize: Figure size as (width, height). If None, auto-calculated
        title: Custom title for the plot
        savefig: Path to save the figure (if None, figure is not saved)
        show: Whether to display the plot
    
    Returns:
        matplotlib figure object if show=False, otherwise None
    """
    if not isinstance(bootstrap_results, pd.DataFrame):
        raise ValueError("bootstrap_results must be a pandas DataFrame")
    
    # Auto-detect comparison groups and metrics
    columns = bootstrap_results.columns.tolist()
    
    # Detect suffixes
    if comparison_col is None:
        # Try to detect common patterns
        if any('_in_sample' in col for col in columns):
            comparison_col = 'sample'
        elif any('_train' in col for col in columns):
            comparison_col = 'train'
        else:
            raise ValueError("Cannot auto-detect comparison groups. Please specify comparison_col")
    
    # Extract unique metrics and groups
    metric_groups = {}
    for col in columns:
        if comparison_col in col.lower():
            # Split by the last occurrence of underscore to separate metric from group
            parts = col.rsplit('_', 2)  # Split from right, max 2 splits
            if len(parts) >= 2:
                metric_base = parts[0]
                group = '_'.join(parts[1:])
                if metric_base not in metric_groups:
                    metric_groups[metric_base] = []
                metric_groups[metric_base].append((col, group))
    
    if not metric_groups:
        raise ValueError(f"No comparison columns found with '{comparison_col}' pattern")
    
    # Filter by requested metrics if specified
    if metrics is not None:
        metric_groups = {k: v for k, v in metric_groups.items() if k in metrics}
        if not metric_groups:
            raise ValueError(f"No matching metrics found: {metrics}")
    
    n_metrics = len(metric_groups)
    
    # Auto-calculate figsize
    if figsize is None:
        cols = min(3, n_metrics)
        rows = int(np.ceil(n_metrics / cols))
        figsize = (6 * cols, 5 * rows)
    
    # Create subplots
    cols = min(3, n_metrics)
    rows = int(np.ceil(n_metrics / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Ensure axes is always a list
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
    
    for idx, (metric_base, group_cols) in enumerate(metric_groups.items()):
        ax = axes[idx]
        
        for color_idx, (col, group) in enumerate(group_cols):
            metric_data = bootstrap_results[col].dropna()
            
            # Calculate statistics
            mean_val = metric_data.mean()
            alpha = 1 - confidence_level
            ci_lower = np.percentile(metric_data, (alpha / 2) * 100)
            ci_upper = np.percentile(metric_data, (1 - alpha / 2) * 100)
            
            # Plot histogram with transparency
            color = colors[color_idx % len(colors)]
            ax.hist(metric_data, bins=30, density=True, alpha=0.5, 
                   color=color, edgecolor='black', linewidth=0.5, 
                   label=f'{group.replace("_", " ").title()}')
            
            # Add mean line
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel(metric_base.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    if title is None:
        title = f'Bootstrap Distribution Comparison: {comparison_col.replace("_", " ").title()}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    # Save if requested
    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    # Show or return
    if show:
        plt.show()
        plt.close()
        return None
    else:
        return fig
