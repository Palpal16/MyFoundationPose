import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define metric properties: direction (↑ higher is better, ↓ lower is better) and units
METRIC_PROPERTIES = {
    'ADD': {'direction': '↑', 'unit': '%', 'higher_is_better': True},
    'ADDS': {'direction': '↑', 'unit': '%', 'higher_is_better': True},
    'ADI': {'direction': '↓', 'unit': 'cm', 'higher_is_better': False},
    '3D_IOU': {'direction': '↑', 'unit': '%', 'higher_is_better': True},
    'CD': {'direction': '↓', 'unit': 'cm', 'higher_is_better': False},
}

# Properties for per-frame plot labels (ADD/ADDS are distances in plots)
PLOT_METRIC_PROPERTIES = {
    'ADD': {'direction': '↓', 'unit': 'cm'},
    'ADDS': {'direction': '↓', 'unit': 'cm'},
    'ADI': {'direction': '↓', 'unit': 'cm'},
    '3D_IOU': {'direction': '↑', 'unit': '%'},
}


def create_summary_table(df, output_path, title, videos=None, summary_metrics=None, current_metric=None):
    """
    Create and save a summary table as both CSV and PNG with bold formatting for max values.

    Parameters:
    - df: DataFrame containing the data
    - output_path: Base path for saving files (without extension)
    - title: Title for the table plot
    - videos: List of videos for determining max values per video (None for single video tables)
    - summary_metrics: List of metrics to check for max values
    - current_metric: The metric name for per-metric tables (used to determine best value direction)
    """
    csv_path = f'{output_path}.csv'
    #df.to_csv(csv_path, index=False, float_format='%.3f')

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')

    # Add arrows and units to column labels on the SAME LINE
    formatted_col_labels = []
    for col in df.columns:
        if col in METRIC_PROPERTIES:
            props = METRIC_PROPERTIES[col]
            formatted_col_labels.append(f"{col} {props['direction']} ({props['unit']})")
        else:
            formatted_col_labels.append(col)

    table = ax.table(cellText=df.round(3).values,
                     colLabels=formatted_col_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Header row (use Polimi blue)
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#5B8999')
        table[(0, i)].set_text_props(weight='bold', color='black')

    # Alternating row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#e1e1e1")

    # Bold maximum/minimum values for each metric
    metrics_to_check = summary_metrics if summary_metrics else [col for col in df.columns if col not in ['Method', 'Video']]

    if metrics_to_check:
        for metric in metrics_to_check:
            if metric in df.columns:
                metric_col_idx = df.columns.get_loc(metric)
                values = df[metric].values
                if not all(np.isnan(values)):
                    # Determine if higher is better
                    if metric in METRIC_PROPERTIES:
                        if METRIC_PROPERTIES[metric]['higher_is_better']:
                            val = np.nanmax(values)
                        else:
                            val = np.nanmin(values)
                    elif current_metric and current_metric in METRIC_PROPERTIES:
                        if METRIC_PROPERTIES[current_metric]['higher_is_better']:
                            val = np.nanmax(values)
                        else:
                            val = np.nanmin(values)
                    else:
                        val = np.nanmin(values)

                    indices = df[df[metric] == val].index.tolist()
                    for idx in indices:
                        table_row = list(df.index).index(idx) + 1
                        table[(table_row, metric_col_idx)].set_text_props(weight='bold')

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    table_plot_path = f'{output_path}.png'
    plt.savefig(table_plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_experiment_results(base_dir, final_base_dir, methods_default, methods_final,
                               videos, plot_metrics, main_metrics, additional_metrics,
                               output_dir='./debug/plots_output'):
    """
    Analyze and visualize experiment results comparing different methods across videos.

    Creates:
    - Line plots for each video showing metrics over frames (all methods compared)
    - Summary tables (CSV and PNG) for each video with mean metrics
    - Metric-specific tables (methods x videos) for each metric
    - Overall summary table with all combinations
    - Mean of means summary table across all videos

    Parameters:
    - base_dir: Base directory containing method folders for methods_default
    - final_base_dir: Base directory for methods_final
    - methods_default: List of methods using base_dir
    - methods_final: List of methods using final_base_dir
    - videos: List of video names
    - plot_metrics: List of per-frame metrics to plot (from metrics.json)
    - main_metrics: List of dict-style metrics in summary (with mean/min/max)
    - additional_metrics: List of scalar metrics in summary (e.g. AUC values)
    - output_dir: Directory to save plots and tables (default: './debug/plots_output')

    Returns:
    - DataFrame with overall summary of all methods and videos
    """
    # Clean output directory if it exists
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"✓ Cleaned output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    methods = methods_default + methods_final
    summary_metrics = main_metrics + additional_metrics

    # Map each method to its base directory
    method_base_dir = {m: base_dir for m in methods_default}
    method_base_dir.update({m: final_base_dir for m in methods_final})

    # Dictionary to store all data
    metrics_data = {}
    summary_data = {}

    # Load all data
    for method in methods:
        metrics_data[method] = {}
        summary_data[method] = {}
        for video in videos:
            metrics_path = os.path.join(method_base_dir[method], method, video, 'evaluation_results', 'metrics.json')
            summary_path = os.path.join(method_base_dir[method], method, video, 'evaluation_results', 'summary.json')

            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data[method][video] = json.load(f)
            else:
                metrics_data[method][video] = None

            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_data[method][video] = json.load(f)
            else:
                summary_data[method][video] = None

    # Create plots for each video
    for video in videos:
        os.makedirs(os.path.join(output_dir, video), exist_ok=True)

        for metric in plot_metrics:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))

            for method in methods:
                if metrics_data[method][video] and metric in metrics_data[method][video]:
                    data = metrics_data[method][video][metric]

                    if metric == '3D_IOU':
                        data = [max(val, 50) for val in data]
                    else:
                        data = [min(val, 0.05) for val in data]

                    frames = range(len(data))
                    linewidth = 3 if method == 'attach' else 2
                    ax.plot(frames, data, label=method, linewidth=linewidth, alpha=0.8)

            ax.set_xlabel('Frame', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            metric_info = f" {PLOT_METRIC_PROPERTIES[metric]['direction']} ({PLOT_METRIC_PROPERTIES[metric]['unit']})" if metric in PLOT_METRIC_PROPERTIES else ""
            ax.set_title(f'Video: {video} - {metric}{metric_info}', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'{video}/{metric}_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

        # Create pairwise comparison plots for each method vs 'attach'
        for method in methods:
            if method == 'attach' or not metrics_data[method][video]:
                continue

            method_folder = os.path.join(output_dir, video, method)
            os.makedirs(method_folder, exist_ok=True)

            for metric in plot_metrics:
                fig, ax = plt.subplots(1, 1, figsize=(16, 8))

                if metrics_data['attach'][video] and metric in metrics_data['attach'][video]:
                    data = metrics_data['attach'][video][metric]

                    if metric == '3D_IOU':
                        data = [max(val, 50) for val in data]
                    else:
                        data = [min(val, 0.05) for val in data]

                    frames = range(len(data))
                    ax.plot(frames, data, label='attach', linewidth=3, alpha=0.8)

                if metric in metrics_data[method][video]:
                    data = metrics_data[method][video][metric]

                    if metric == '3D_IOU':
                        data = [max(val, 50) for val in data]
                    else:
                        data = [min(val, 0.05) for val in data]

                    frames = range(len(data))
                    ax.plot(frames, data, label=method, linewidth=2, alpha=0.8)

                ax.set_xlabel('Frame', fontsize=11)
                ax.set_ylabel(metric, fontsize=11)
                metric_info = f" {PLOT_METRIC_PROPERTIES[metric]['direction']} ({PLOT_METRIC_PROPERTIES[metric]['unit']})" if metric in PLOT_METRIC_PROPERTIES else ""
                ax.set_title(f'Video: {video} - {metric}{metric_info} ({method} vs attach)', fontsize=12, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_path = os.path.join(method_folder, f'{metric}_comparison.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

    # Create summary tables for each video
    for video in videos:
        table_data = []
        for method in methods:
            row = {'Method': method}
            if summary_data[method][video]:
                for metric in summary_metrics:
                    if metric in additional_metrics:
                        row[metric] = summary_data[method][video].get(metric, np.nan)
                    else:
                        metric_data = summary_data[method][video].get(metric, {})
                        row[metric] = metric_data.get('mean', np.nan) if isinstance(metric_data, dict) else np.nan
            else:
                for metric in summary_metrics:
                    row[metric] = np.nan
            table_data.append(row)

        df = pd.DataFrame(table_data)
        output_path = os.path.join(output_dir, f'{video}/summary_table')
        title = f'{video} - Mean Metrics Summary'
        create_summary_table(df, output_path, title, videos=None, summary_metrics=summary_metrics)

    # Create overall comparison table ordered by video
    overall_data = []
    for video in videos:
        for method in methods:
            row = {'Video': video, 'Method': method}
            if summary_data[method][video]:
                for metric in summary_metrics:
                    if metric in additional_metrics:
                        row[metric] = summary_data[method][video].get(metric, np.nan)
                    else:
                        metric_data = summary_data[method][video].get(metric, {})
                        row[metric] = metric_data.get('mean', np.nan) if isinstance(metric_data, dict) else np.nan
            else:
                for metric in summary_metrics:
                    row[metric] = np.nan
            overall_data.append(row)

    overall_df = pd.DataFrame(overall_data)

    # Create mean of means summary table
    mean_of_means_data = []
    for method in methods:
        row = {'Method': method}
        for metric in summary_metrics:
            metric_values = [overall_df[(overall_df['Method'] == method) & (overall_df['Video'] == video)][metric].values[0]
                           for video in videos if len(overall_df[(overall_df['Method'] == method) & (overall_df['Video'] == video)][metric].values) > 0]
            row[metric] = np.nanmean(metric_values) if metric_values else np.nan
        mean_of_means_data.append(row)

    mean_of_means_df = pd.DataFrame(mean_of_means_data)
    mean_output_path = os.path.join(output_dir, 'overall_averages_table')
    create_summary_table(mean_of_means_df, mean_output_path, 'Average Across All Videos', videos=None, summary_metrics=summary_metrics)

    # Create metric-specific tables (methods x videos)
    for metric in summary_metrics:
        metric_table_data = []
        for method in methods:
            row = {'Method': method}
            for video in videos:
                if summary_data[method][video]:
                    if metric in additional_metrics:
                        row[video] = summary_data[method][video].get(metric, np.nan)
                    else:
                        metric_data = summary_data[method][video].get(metric, {})
                        row[video] = metric_data.get('mean', np.nan) if isinstance(metric_data, dict) else np.nan
                else:
                    row[video] = np.nan
            metric_table_data.append(row)

        metric_df = pd.DataFrame(metric_table_data)
        metric_output_path = os.path.join(output_dir, f'metric_table_{metric.replace("(", "").replace(")", "").replace("-", "_")}')
        metric_title = f'{metric} {METRIC_PROPERTIES.get(metric, {}).get("direction", "")} ({METRIC_PROPERTIES.get(metric, {}).get("unit", "")}) - Methods vs Videos'
        create_summary_table(metric_df, metric_output_path, metric_title, videos=None, summary_metrics=None, current_metric=metric)

    return overall_df


# Usage example:
if __name__ == "__main__":
    base_dir = '/Experiments/simonep01/Results'
    final_base_dir = './debug'

    methods_default = ['fp', 'sam3d', 'ua_psnr', 'any6d']
    methods_final = ['attach']

    videos = ['AP10', 'AP11', 'AP12', 'AP13', 'AP14', 'MPM10', 'MPM11', 'MPM12', 'MPM13', 'MPM14', 'SB11', 'SB13', 'SM1']
    plot_metrics = ['ADD', 'ADDS', 'ADI', '3D_IOU']
    main_metrics = ['ADI', '3D_IOU']
    additional_metrics = ['ADD', 'ADDS', 'CD']

    results = analyze_experiment_results(
        base_dir=base_dir,
        final_base_dir=final_base_dir,
        methods_default=methods_default,
        methods_final=methods_final,
        videos=videos,
        plot_metrics=plot_metrics,
        main_metrics=main_metrics,
        additional_metrics=additional_metrics,
        output_dir=f'{final_base_dir}/plots_output'
    )
