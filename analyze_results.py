import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def create_summary_table(df, output_path, title, videos=None, summary_metrics=None):
    """
    Create and save a summary table as both CSV and PNG with bold formatting for max values.

    Parameters:
    - df: DataFrame containing the data
    - output_path: Base path for saving files (without extension)
    - title: Title for the table plot
    - videos: List of videos for determining max values per video (None for single video tables)
    - summary_metrics: List of metrics to check for max values
    """
    csv_path = f'{output_path}.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.round(4).values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    # Bold maximum values for each video and metric
    if summary_metrics:
        for metric in summary_metrics:
            if metric in df.columns:
                metric_col_idx = df.columns.get_loc(metric)
                values = df[metric].values
                if not all(np.isnan(values)):
                    if metric in ['3D_IOU', 'ADD(S)-0.1']:
                        val = np.nanmax(values)
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


def analyze_experiment_results(base_dir='./debug', output_dir='./debug/plots_output'):
    """
    Analyze and visualize experiment results comparing different methods across videos.

    Creates:
    - Line plots for each video showing metrics over frames (4 methods compared)
    - Summary tables (CSV and PNG) for each video with mean metrics
    - Bar plots comparing each metric across all videos
    - Overall summary table with all combinations

    Parameters:
    - base_dir: Base directory containing method folders (default: './debug')
    - output_dir: Directory to save plots and tables (default: './plots_output')

    Returns:
    - DataFrame with overall summary of all methods and videos
    """
    os.makedirs(output_dir, exist_ok=True)

    methods = ['fp', 'attach', 'sam3d', 'any6d']
    videos = ['AP14', 'SM1', 'SB13', 'MPM10']
    main_metrics = ['ADD(S)', 'ADI', '3D_IOU']
    additional_metrics = ['CD', 'ADD(S)-0.1']
    summary_metrics = main_metrics + additional_metrics

    # Dictionary to store all data
    metrics_data = {}
    summary_data = {}

    # Load all data
    for method in methods:
        metrics_data[method] = {}
        summary_data[method] = {}
        for video in videos:
            metrics_path = os.path.join(base_dir, method, video, 'evaluation_results', 'metrics.json')
            summary_path = os.path.join(base_dir, method, video, 'evaluation_results', 'summary.json')

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

        for metric in main_metrics:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))

            for method in methods:
                if metrics_data[method][video] and metric in metrics_data[method][video]:
                    data = metrics_data[method][video][metric]
                    if metric == '3D_IOU':
                        data = [max(val, 50) for val in data]
                    else:
                        data = [min(val, 0.05) for val in data]
                    frames = range(len(data))

                    # Use bigger line width for 'attach' method
                    linewidth = 3 if method == 'attach' else 2
                    ax.plot(frames, data, label=method, linewidth=linewidth, alpha=0.8)

            ax.set_xlabel('Frame', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'Video: {video} - {metric}', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f'{video}/{metric}_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

    # Create summary tables for each video using the new function
    for video in videos:
        table_data = []
        for method in methods:
            row = {'Method': method}
            if summary_data[method][video]:
                for metric in summary_metrics:
                    if metric == 'CD':
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
                    if metric == 'CD':
                        row[metric] = summary_data[method][video].get(metric, np.nan)
                    else:
                        metric_data = summary_data[method][video].get(metric, {})
                        row[metric] = metric_data.get('mean', np.nan) if isinstance(metric_data, dict) else np.nan
            else:
                for metric in summary_metrics:
                    row[metric] = np.nan
            overall_data.append(row)

    overall_df = pd.DataFrame(overall_data)
    overall_output_path = os.path.join(output_dir, 'overall_summary_table')

    return overall_df


# Usage example:
if __name__ == "__main__":
    results = analyze_experiment_results(base_dir='./debug', output_dir='./debug/plots_output')
