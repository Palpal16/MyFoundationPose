import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define metric properties: direction (↑ higher is better, ↓ lower is better) and units
METRIC_PROPERTIES = {
    'ADD(S)': {'direction': '↓', 'unit': 'cm', 'higher_is_better': False},
    'ADI': {'direction': '↓', 'unit': 'cm', 'higher_is_better': False},
    '3D_IOU': {'direction': '↑', 'unit': '%', 'higher_is_better': True},
    'CD': {'direction': '↓', 'unit': 'cm', 'higher_is_better': False},
    'ADD(S)-0.1': {'direction': '↑', 'unit': '%', 'higher_is_better': True}
}

def create_summary_table(df, csv_output_path, png_output_path, title, summary_metrics=None, current_metric=None, separate_fp=True):
    """
    Create and save a summary table as both CSV and PNG with bold formatting for max values.
    FP method is renamed as 'Baseline' and placed at bottom with Mean and Median rows.

    Parameters:
    - df: DataFrame containing the data
    - csv_output_path: Full path for saving CSV file
    - png_output_path: Full path for saving PNG file
    - title: Title for the table plot
    - summary_metrics: List of metrics to check for max values
    - current_metric: The metric name for per-metric tables (used to determine best value direction)
    - separate_fp: Whether to separate FP row at the bottom (default: True)
    """
    # Save CSV before reordering (without modifications)
    df.to_csv(csv_output_path, index=False, float_format='%.3f')
    
    # Reorder dataframe for PNG: move 'fp' to bottom and add mean/median if it exists
    df_display = df.copy()
    if separate_fp and 'Method' in df_display.columns:
        fp_rows = df_display[df_display['Method'] == 'fp'].copy()
        other_rows = df_display[df_display['Method'] != 'fp'].copy()
        
        if not fp_rows.empty and not other_rows.empty:
            # Calculate mean and median for the 4 methods (excluding fp)
            mean_row = {'Method': 'Mean'}
            median_row = {'Method': 'Median'}
            
            for col in df_display.columns:
                if col != 'Method':
                    values = pd.to_numeric(other_rows[col], errors='coerce')
                    mean_row[col] = np.nanmean(values)
                    median_row[col] = np.nanmedian(values)
            
            # Rename fp to Baseline
            fp_rows.loc[:, 'Method'] = 'Baseline'
            
            # Combine: other methods, then mean, median, baseline
            mean_df = pd.DataFrame([mean_row])
            median_df = pd.DataFrame([median_row])
            df_display = pd.concat([other_rows, mean_df, median_df, fp_rows], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, max(3, len(df_display) * 0.3 + 0.5)))
    ax.axis('tight')
    ax.axis('off')

    # Add arrows and units to column labels
    formatted_col_labels = []
    for col in df_display.columns:
        if col in METRIC_PROPERTIES:
            props = METRIC_PROPERTIES[col]
            formatted_col_labels.append(f"{col} {props['direction']} ({props['unit']})")
        else:
            formatted_col_labels.append(col)

    # Round to 3 decimal places for display
    df_rounded = df_display.copy()
    for col in df_rounded.columns:
        if col not in ['Method', 'Video']:
            df_rounded[col] = df_rounded[col].apply(lambda x: f'{float(x):.3f}' if pd.notna(x) else x)

    table = ax.table(cellText=df_rounded.values,
                     colLabels=formatted_col_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Header row (use Polimi blue)
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#5B8999')
        table[(0, i)].set_text_props(weight='bold', color='black')

    # Alternating row colors
    for i in range(1, len(df_display) + 1):
        for j in range(len(df_display.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#e1e1e1")

    # Add double line separator between other_rows and Mean/Median/Baseline
    if separate_fp and 'Method' in df_display.columns:
        mean_row_idx = df_display[df_display['Method'] == 'Mean'].index
        if not mean_row_idx.empty:
            separator_position = mean_row_idx[0] + 1  # Position in table (row index + 1 for header)
            
            # Hide the bottom edge of the row BEFORE the separator (last other_row)
            for j in range(len(df_display.columns)):
                table[(separator_position - 1, j)].visible_edges = 'TLR'  # Top, Left, Right only (no Bottom)
                table[(separator_position, j)].visible_edges = 'BLR'  # No top edge for Mean row
            
            # Calculate y-position for the double line in figure coordinates
            total_rows = len(df_display) + 1  # +1 for header
            y_pos = (total_rows - separator_position) / total_rows
            
            # Draw two parallel horizontal lines
            line_spacing = 0.006
            ax.plot([0, 1], [y_pos + line_spacing, y_pos + line_spacing], 
                    transform=ax.transAxes, color='black', linewidth=1.1, clip_on=False)
            ax.plot([0, 1], [y_pos - line_spacing, y_pos - line_spacing], 
                    transform=ax.transAxes, color='black', linewidth=1.1, clip_on=False)




    # Bold maximum/minimum values for each metric (EXCLUDING fp/baseline, mean, median rows)
    metrics_to_check = summary_metrics if summary_metrics else [col for col in df.columns if col not in ['Method', 'Video']]

    if metrics_to_check:
        # Create a filtered dataframe excluding 'fp', 'Mean', 'Median', 'Baseline' for finding best values
        if 'Method' in df.columns:
            df_filtered = df[(df['Method'] != 'fp') & (df['Method'] != '')]
        else:
            df_filtered = df[df.iloc[:, 0] != '']
        
        for metric in metrics_to_check:
            if metric in df.columns and not df_filtered.empty:
                metric_col_idx = df.columns.get_loc(metric)
                values = df_filtered[metric].values
                
                # Convert to numeric, handling any non-numeric values
                values = pd.to_numeric(values, errors='coerce')
                
                if not all(np.isnan(values)):
                    # Determine if higher is better
                    if metric in METRIC_PROPERTIES:
                        if METRIC_PROPERTIES[metric]['higher_is_better']:
                            best_val = np.nanmax(values)
                        else:
                            best_val = np.nanmin(values)
                    elif current_metric and current_metric in METRIC_PROPERTIES:
                        if METRIC_PROPERTIES[current_metric]['higher_is_better']:
                            best_val = np.nanmax(values)
                        else:
                            best_val = np.nanmin(values)
                    else:
                        best_val = np.nanmin(values)

                    # Find indices in filtered dataframe
                    filtered_indices = df_filtered[df_filtered[metric] == best_val].index.tolist()
                    
                    # Map back to display dataframe indices and bold those cells
                    for orig_idx in filtered_indices:
                        # Find the position in df_display
                        if 'Method' in df.columns:
                            method_name = df.loc[orig_idx, 'Method']
                            matching_rows = df_display[df_display['Method'] == method_name]
                            if not matching_rows.empty:
                                display_idx = matching_rows.index[0]
                                table_row = list(df_display.index).index(display_idx) + 1
                                table[(table_row, metric_col_idx)].set_text_props(weight='bold')

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.savefig(png_output_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_experiment_results(base_dir='./debug', output_dir='./debug/tables_output'):
    """
    Analyze experiment results and create summary tables comparing different methods.

    Creates:
    - Metric-specific tables (methods x videos) for each metric
    - Overall summary table with mean of means across all videos

    Parameters:
    - base_dir: Base directory containing method folders (default: './debug')
    - output_dir: Directory to save tables (default: './debug/tables_output')

    Returns:
    - DataFrame with overall summary of all methods and videos
    """
    # Clean output directory if it exists
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"✓ Cleaned output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subfolders for CSVs and PNGs
    csv_dir = os.path.join(output_dir, 'csv')
    png_dir = os.path.join(output_dir, 'png')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Updated method order: ua before any6d
    methods = ['fp', 'attach', 'final', 'sam3d', 'ua', 'any6d']
    videos = ['AP10', 'AP11', 'AP12', 'AP13', 'AP14', 'MPM10', 'MPM11', 'MPM12', 'MPM13', 'MPM14', 'SB11', 'SB13', 'SM1']
    main_metrics = ['ADD(S)', 'ADI', '3D_IOU']
    additional_metrics = ['CD', 'ADD(S)-0.1']
    summary_metrics = main_metrics + additional_metrics

    # Dictionary to store summary data
    summary_data = {}

    # Load all summary data
    for method in methods:
        summary_data[method] = {}
        for video in videos:
            summary_path = os.path.join(base_dir, method, video, 'evaluation_results', 'summary.json')
            if method == 'final':
                summary_path = os.path.join('./debug', method, video, 'evaluation_results', 'summary.json')

            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_data[method][video] = json.load(f)
            else:
                summary_data[method][video] = None

    # Create overall comparison data
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
    csv_path = os.path.join(csv_dir, 'overall_averages_table.csv')
    png_path = os.path.join(png_dir, 'overall_averages_table.png')
    create_summary_table(mean_of_means_df, csv_path, png_path, 'Average Across All Videos', 
                        summary_metrics=summary_metrics, separate_fp=True)
    print(f"✓ Created overall averages table")

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
        metric_filename = f'metric_table_{metric.replace("(", "").replace(")", "").replace("-", "_")}'
        csv_path = os.path.join(csv_dir, f'{metric_filename}.csv')
        png_path = os.path.join(png_dir, f'{metric_filename}.png')
        metric_title = f'{metric} {METRIC_PROPERTIES.get(metric, {}).get("direction", "")} ({METRIC_PROPERTIES.get(metric, {}).get("unit", "")}) - Methods vs Videos'
        create_summary_table(metric_df, csv_path, png_path, metric_title, 
                           summary_metrics=None, current_metric=metric, separate_fp=True)
        print(f"✓ Created metric table for {metric}")

    return overall_df

# Usage example:
if __name__ == "__main__":
    results = analyze_experiment_results(base_dir='/Experiments/simonep01/Results', output_dir='./debug/tables_output')
    print("\n✓ All tables generated successfully!")
    print(f"✓ CSVs saved in: debug/tables_output/csv/")
    print(f"✓ PNGs saved in: debug/tables_output/png/")