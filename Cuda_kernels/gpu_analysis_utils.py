import pandas as pd
import numpy as np
from scipy import stats

def load_and_preprocess_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Convert columns to appropriate numeric types
    numeric_columns = [
        'smFrequencyCyclePerUsecond', 'elapsedCycles', 'memoryThroughputPercent',
        'dramThroughputPercent', 'durationUsecond', 'l1TexCacheThroughputPercent',
        'l2CacheThroughputPercent', 'smActiveCycles', 'computeSMThroughputPercent',
        'registersPerThread', 'sharedMemoryConfigSizeKbyte', 'driverSharedMemoryPerBlockByte',
        'dynamicSharedMemoryPerBlockKbyte', 'staticSharedMemoryPerBlockByte', 'wavesPerSM',
        'blockLimitSm', 'blockLimitRegisters', 'blockLimitSharedMem', 'blockLimitWarps',
        'theoreticalActiveWarpsPerSm', 'theoreticalOccupancyPercent', 'achievedOccupancyPercent',
        'achievedActiveWarpsPerSm'
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle 'N/A' values
    df = df.replace('N/A', np.nan)

    # Create block_config column
    df['block_config'] = df['blockSizeX'].astype(str) + 'x' + df['blockSizeY'].astype(str)
    df['total_threads'] = df['blockSizeX'] * df['blockSizeY']

    return df

def calculate_kpi(df):
    # Existing KPIs
    df['SM Efficiency'] = df['computeSMThroughputPercent'] / (df['smFrequencyCyclePerUsecond'] * df['elapsedCycles'])
    df['Compute to Memory Ratio'] = df['smActiveCycles'] / df['elapsedCycles']
    df['DRAM vs SM Frequency'] = df['dramThroughputPercent'] / df['smFrequencyCyclePerUsecond']
    df['Cache Hit Rate'] = (df['l1TexCacheThroughputPercent'] + df['l2CacheThroughputPercent']) / df['memoryThroughputPercent']
    df['Occupancy'] = df['achievedActiveWarpsPerSm'] / df['theoreticalActiveWarpsPerSm']
    
    # New derived metrics
    df['Execution Time Per Thread'] = df['durationUsecond'] / df['total_threads']
    df['Memory Efficiency'] = df['memoryThroughputPercent'] / df['durationUsecond']
    df['Compute/Memory Efficiency Ratio'] = df['computeSMThroughputPercent'] / df['memoryThroughputPercent']
    df['Normalized Occupancy'] = df['achievedOccupancyPercent'] / 100
    df['Normalized SM Efficiency'] = df['SM Efficiency'] / df['theoreticalOccupancyPercent']
    
    return df


def format_value(value):
    """Format value to handle large and small numbers."""
    if np.isnan(value):
        return "N/A"
    if abs(value) < 0.01 or abs(value) >= 1e5:
        return f"{value:.4e}"
    else:
        return f"{value:.4f}"

def find_optimal_config(df, metric, maximize=True):
    if maximize:
        optimal_row = df.loc[df[metric].idxmax()]
    else:
        optimal_row = df.loc[df[metric].idxmin()]
    return optimal_row['block_config'], optimal_row[metric]

def generate_graph_analysis(df, x_axis, y_axis, metric_descriptions):
    """Generate a detailed analysis for a specific graph."""
    analysis = f"Graph Analysis: {x_axis} vs {y_axis}\n\n"

    # Determine if we want to minimize or maximize the metric
    metrics_to_minimize = ['durationUsecond', 'elapsedCycles', 'Execution Time Per Thread', 'Performance Variability']
    maximize = y_axis not in metrics_to_minimize

    # Find optimal and worst configurations
    optimal_config, optimal_value = find_optimal_config(df, y_axis, maximize=maximize)
    worst_config, worst_value = find_optimal_config(df, y_axis, maximize=not maximize)

    # Calculate overall statistics
    mean_value = df[y_axis].mean()
    std_value = df[y_axis].std()

    analysis += f"Optimal configuration: {optimal_config} with {y_axis} value of {format_value(optimal_value)}\n"
    analysis += f"Worst configuration: {worst_config} with {y_axis} value of {format_value(worst_value)}\n"
    analysis += f"Mean {y_axis}: {format_value(mean_value)}\n"
    analysis += f"Standard deviation of {y_axis}: {format_value(std_value)}\n\n"

    if y_axis in metric_descriptions:
        analysis += f"Metric Description: {metric_descriptions[y_axis]}\n\n"

    improvement = abs(optimal_value - worst_value) / abs(worst_value) * 100

    if y_axis in ['durationUsecond', 'Execution Time Per Thread']:
        analysis += f"The optimal block size {optimal_config} achieves the lowest execution time of {format_value(optimal_value)} microseconds. "
        analysis += f"This is {format_value(improvement)}% faster than the worst-performing configuration ({worst_config}).\n"
        analysis += "Lower execution times indicate better performance.\n"

    elif y_axis in ['SM Efficiency', 'Compute Throughput', 'Memory Throughput', 'Occupancy', 'Warp Execution Efficiency']:
        analysis += f"The block size {optimal_config} achieves the highest {y_axis} of {format_value(optimal_value)}. "
        analysis += f"This is {format_value(improvement)}% better than the worst-performing configuration ({worst_config}).\n"
        analysis += f"Higher {y_axis} generally indicates better GPU resource utilization and performance.\n"

    elif y_axis == 'Compute to Memory Ratio':
        analysis += f"The block size {optimal_config} achieves the best compute to memory ratio of {format_value(optimal_value)}. "
        if optimal_value > 1:
            analysis += "This kernel appears to be compute-bound, spending more time on computations than memory operations.\n"
        else:
            analysis += "This kernel appears to be memory-bound, spending more time on memory operations than computations.\n"

    elif y_axis in ['Cache Hit Rate', 'L1 Cache Throughput', 'L2 Cache Throughput']:
        analysis += f"The block size {optimal_config} achieves the highest {y_axis} of {format_value(optimal_value)}. "
        analysis += f"This is {format_value(improvement)}% better than the worst-performing configuration ({worst_config}).\n"
        analysis += "Higher cache hit rates and throughputs indicate better utilization of the cache memory, potentially reducing memory access times.\n"

    elif y_axis in ['Shared Memory Usage', 'Dynamic Shared Memory Usage']:
        analysis += f"The block size {optimal_config} uses {format_value(optimal_value)} KB of shared memory. "
        analysis += "Higher shared memory usage can improve performance for certain algorithms, but excessive usage can limit occupancy.\n"

    elif y_axis == 'Registers Per Thread':
        analysis += f"The block size {optimal_config} uses {format_value(optimal_value)} registers per thread. "
        analysis += "Higher register usage can reduce occupancy but may be necessary for complex computations.\n"

    elif y_axis == 'Performance Variability':
        analysis += f"The block size {optimal_config} shows the lowest performance variability of {format_value(optimal_value)}. "
        analysis += f"This is {format_value(improvement)}% more consistent than the worst-performing configuration ({worst_config}).\n"
        analysis += "Lower performance variability indicates more consistent execution times across runs.\n"

    else:
        if maximize:
            analysis += f"The block size {optimal_config} achieves the highest {y_axis} of {format_value(optimal_value)}. "
        else:
            analysis += f"The block size {optimal_config} achieves the lowest {y_axis} of {format_value(optimal_value)}. "
        analysis += f"This is {format_value(improvement)}% better than the worst-performing configuration ({worst_config}).\n"

    return analysis

# ... (other functions in the file remain the same)

def generate_analysis(df, metric):
    optimal_config, optimal_value = find_optimal_config(df, metric, maximize=(metric != 'durationUsecond'))
    mean_value = df[metric].mean()
    std_value = df[metric].std()
    
    analysis = f"Analysis for {metric}:\n"
    analysis += f"Optimal block configuration: {optimal_config}\n"
    analysis += f"Optimal value: {format_value(optimal_value)}\n"
    analysis += f"Mean value: {format_value(mean_value)}\n"
    analysis += f"Standard deviation: {format_value(std_value)}\n\n"
    
    if metric == 'durationUsecond':
        analysis += f"The optimal block configuration of {optimal_config} achieves the lowest execution time of {format_value(optimal_value)} microseconds. "
        analysis += f"This is {format_value(((mean_value - optimal_value) / mean_value * 100))}% faster than the average execution time across all configurations."
    elif metric == 'SM Efficiency':
        analysis += f"The block configuration {optimal_config} achieves the highest SM efficiency of {format_value(optimal_value)}. "
        analysis += f"This is {format_value(((optimal_value - mean_value) / mean_value * 100))}% better than the average SM efficiency across all configurations."
    elif metric == 'Compute to Memory Ratio':
        analysis += f"The optimal block configuration {optimal_config} achieves a compute to memory ratio of {format_value(optimal_value)}. "
        analysis += f"This indicates a good balance between computation and memory operations, with computation taking {format_value(optimal_value)} times as long as memory operations."
    elif metric == 'Occupancy':
        analysis += f"The block configuration {optimal_config} achieves the highest occupancy of {format_value(optimal_value)}. "
        analysis += f"This indicates that this configuration utilizes {format_value(optimal_value * 100)}% of the GPU's theoretical maximum occupancy."
    
    return analysis



def get_trend_analysis(df, x_axis, y_axis):
    if df[x_axis].dtype == 'object' or df[y_axis].dtype == 'object':
        # Handle categorical data
        if df[x_axis].dtype == 'object':
            cat_column, num_column = x_axis, y_axis
        else:
            cat_column, num_column = y_axis, x_axis
        
        group_means = df.groupby(cat_column)[num_column].mean().sort_values(ascending=False)
        analysis = f"Trend analysis for {x_axis} vs {y_axis}:\n\n"
        analysis += f"Top 3 {cat_column} categories for {num_column}:\n"
        for i, (category, mean_value) in enumerate(group_means.head(3).items(), 1):
            analysis += f"{i}. {category}: {format_value(mean_value)}\n"
        
        analysis += f"\nBottom 3 {cat_column} categories for {num_column}:\n"
        for i, (category, mean_value) in enumerate(group_means.tail(3).items(), 1):
            analysis += f"{i}. {category}: {format_value(mean_value)}\n"
        
        # Add overall statistics
        # overall_mean = df[num_column].mean()
        # overall_std = df[num_column].std()
        # analysis += f"\nOverall mean: {format_value(overall_mean)}\n"
        # analysis += f"Overall standard deviation: {format_value(overall_std)}\n"
        
        # Uncomment the following lines if you want to include ANOVA test
        # f_statistic, p_value = stats.f_oneway(*[group.values for name, group in df.groupby(cat_column)[num_column]])
        # analysis += f"\nANOVA test p-value: {format_value(p_value)}\n"
        # if p_value < 0.05:
        #     analysis += f"There is a statistically significant difference in {num_column} across different {cat_column} categories."
        # else:
        #     analysis += f"There is no statistically significant difference in {num_column} across different {cat_column} categories."
    
    else:
        # Handle numeric data
        correlation = df[x_axis].corr(df[y_axis])
        analysis = f"Trend analysis for {x_axis} vs {y_axis}:\n\n"
        analysis += f"Correlation coefficient: {format_value(correlation)}\n\n"
        
        if abs(correlation) < 0.3:
            analysis += f"There appears to be a weak relationship between {x_axis} and {y_axis}. "
            analysis += "This suggests that changes in one variable have little effect on the other."
        elif abs(correlation) < 0.7:
            analysis += f"There appears to be a moderate relationship between {x_axis} and {y_axis}. "
            if correlation > 0:
                analysis += f"As {x_axis} increases, {y_axis} tends to increase as well."
            else:
                analysis += f"As {x_axis} increases, {y_axis} tends to decrease."
        else:
            analysis += f"There appears to be a strong relationship between {x_axis} and {y_axis}. "
            if correlation > 0:
                analysis += f"As {x_axis} increases, {y_axis} strongly tends to increase as well."
            else:
                analysis += f"As {x_axis} increases, {y_axis} strongly tends to decrease."
    
    return analysis
