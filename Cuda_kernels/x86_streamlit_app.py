import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from gpu_analysis_utils import load_and_preprocess_data, calculate_kpi, generate_graph_analysis, get_trend_analysis, find_optimal_config,generate_graph_analysis, generate_analysis


# Load the dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['block_config'] = df['blockSizeX'].astype(str) + 'x' + df['blockSizeY'].astype(str)
    df['total_threads'] = df['blockSizeX'] * df['blockSizeY']
    return df

# Take CSV path from CLI argument
data = load_data(sys.argv[1])

# Convert relevant columns to numeric
numeric_columns = [
    'elapsedCycles', 'smFrequencyCyclePerUsecond', 'memoryThroughputPercent',
    'dramThroughputPercent', 'durationUsecond', 'l1TexCacheThroughputPercent',
    'l2CacheThroughputPercent', 'smActiveCycles', 'computeSMThroughputPercent',
    'registersPerThread', 'sharedMemoryConfigSizeKbyte', 'driverSharedMemoryPerBlockByte',
    'dynamicSharedMemoryPerBlockKbyte', 'staticSharedMemoryPerBlockByte', 'wavesPerSM',
    'blockLimitSm', 'blockLimitRegisters', 'blockLimitSharedMem', 'blockLimitWarps',
    'theoreticalActiveWarpsPerSm', 'theoreticalOccupancyPercent', 'achievedOccupancyPercent',
    'achievedActiveWarpsPerSm'
]

data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Sidebar configuration for selecting block sizes
st.sidebar.header("Block Size Configuration")
min_x, max_x = st.sidebar.select_slider(
    'Select Block Size X Range',
    options=sorted(data['blockSizeX'].unique()),
    value=(min(data['blockSizeX']), max(data['blockSizeX']))
)

min_y, max_y = st.sidebar.select_slider(
    'Select Block Size Y Range',
    options=sorted(data['blockSizeY'].unique()),
    value=(min(data['blockSizeY']), max(data['blockSizeY']))
)

# Filter the data based on the selected ranges
filtered_data = data[(data['blockSizeX'] >= min_x) & (data['blockSizeX'] <= max_x) &
                     (data['blockSizeY'] >= min_y) & (data['blockSizeY'] <= max_y)]

# Kernel name selection
kernel_name = st.sidebar.selectbox('Select Kernel Name', data['kernelName'].unique())
filtered_data = filtered_data[filtered_data['kernelName'] == kernel_name]

def scatter_plot(x, y, size=None, title=None, x_label=None, y_label=None, size_label=None):
    if size:
        filtered_data[size] = filtered_data[size].fillna(filtered_data[size].median())
        fig = px.scatter(
            filtered_data, x=x, y=y, size=size, hover_name='block_config',
            title=title, labels={x: x_label, y: y_label, size: size_label},
            color_continuous_scale=px.colors.sequential.Viridis
        )
    else:
        fig = px.scatter(
            filtered_data, x=x, y=y, hover_name='block_config',
            title=title, labels={x: x_label, y: y_label}
        )

    # Add statistical annotations
    optimal_config, optimal_value = find_optimal_config(filtered_data, y, maximize=(y != 'durationUsecond'))
    mean_value = filtered_data[y].mean()
    std_value = filtered_data[y].std()

    annotations = [
        dict(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"Optimal config: {optimal_config}<br>Optimal value: {optimal_value:.4e}<br>Mean: {mean_value:.4e}<br>Std Dev: {std_value:.4e}",
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        )
    ]

    fig.update_layout(annotations=annotations)

    st.plotly_chart(fig)
    

    
    # Add the new graph-specific analysis
    st.write(generate_graph_analysis(filtered_data, x, y, metric_descriptions))





# Function to create interactive histograms using Plotly
def histogram(column, title=None, x_label=None):
    fig = px.histogram(filtered_data, x=column, nbins=50, title=title, labels={column: x_label})
    st.plotly_chart(fig)

# Function to create interactive 3D scatter plots using Plotly
def scatter_3d(x, y, z, title=None, x_label=None, y_label=None, z_label=None):
    fig = px.scatter_3d(
        filtered_data, x=x, y=y, z=z, color='block_config', title=title,
        labels={x: x_label, y: y_label, z: z_label}
    )
    st.plotly_chart(fig)


metric_descriptions = {
    # Basic Metrics
    'block_config': 'Block Configuration - This refers to the combination of block size in the x and y dimensions. It determines the number of threads that will run concurrently within a block.',
    'durationUsecond': 'Execution Time (µs) - This is the time taken for the kernel to execute. Lower values indicate faster execution, which is generally desirable.',
    'elapsedCycles': 'Elapsed Cycles - The total number of cycles taken for the kernel execution. Lower values indicate faster execution.',
    'gridSize': 'Grid Size - The total number of blocks in the grid along each dimension. Affects the parallelism and efficiency of kernel execution.',
    
    # Derived Metrics
    'SM Efficiency': 'SM Efficiency = Compute throughput divided by the product of SM frequency and elapsed cycles. Higher values indicate more efficient use of the Streaming Multiprocessors.',
    'Compute to Memory Ratio': 'Compute to Memory Ratio = SM active cycles divided by elapsed cycles. It indicates the balance between computation and memory access. Higher values mean more compute-bound performance.',
    'DRAM vs SM Frequency': 'DRAM vs SM Frequency = DRAM throughput divided by SM frequency. It shows how well the memory bandwidth matches the compute capability.',
    'Cache Hit Rate': 'Cache Hit Rate = (L1 + L2 Cache Throughput) divided by Memory Throughput. Higher values indicate that more data is being accessed from the cache, reducing the need to access slower DRAM.',
    'Occupancy': 'Occupancy (%) - This is the ratio of active warps to the maximum possible warps on an SM. Higher values indicate better utilization of GPU resources.',
    'Warp Execution Efficiency': 'Warp Execution Efficiency = Achieved active warps divided by theoretical active warps. Higher values indicate that warps are being utilized effectively.',
    'Compute Throughput': 'Compute Throughput (%) - This shows how much of the compute capability of the GPU is being utilized. Higher values indicate better utilization of the compute resources.',
    'Memory Throughput': 'Memory Throughput (%) - Shows the percentage of maximum possible memory throughput utilized by the kernel. Higher values indicate better memory bandwidth utilization.',
    'L1 Cache Throughput': 'L1 Cache Throughput (%) - Represents the data throughput achieved by the L1 cache. Higher values indicate more efficient use of the L1 cache.',
    'L2 Cache Throughput': 'L2 Cache Throughput (%) - Represents the data throughput achieved by the L2 cache. Higher values indicate more efficient use of the L2 cache.',
    'Achieved Occupancy': 'Achieved Occupancy (%) - The actual occupancy of the kernel compared to the theoretical maximum. Higher values indicate better utilization of the GPU resources.',
    'Compute SM Throughput': 'Compute SM Throughput (%) - The throughput achieved by the streaming multiprocessors. Higher values indicate better utilization of the compute units.',
    'DRAM Throughput': 'DRAM Throughput (%) - The throughput achieved by the DRAM. Higher values indicate better utilization of the memory bandwidth.',
    'Shared Memory Usage': 'Shared Memory Usage (KB) - The amount of shared memory used per block. Higher values indicate more shared memory is used, which can reduce the number of blocks that can run concurrently.',
    'Registers Per Thread': 'Registers Per Thread - The number of registers used by each thread. Higher values can reduce occupancy but may be necessary for complex computations.',
    'Block Limit Warps': 'Block Limit Warps - The maximum number of warps that can be active per block. Higher values indicate more concurrent threads.',
    'Block Limit Shared Memory': 'Block Limit Shared Memory (KB) - The maximum shared memory available per block. Higher values allow more data to be stored on-chip, reducing global memory accesses.',
    'Performance Variability': 'Performance Variability - The standard deviation of execution time divided by the mean execution time. Lower values indicate more consistent performance across runs.',
    'Execution Time Per Thread': 'Execution Time Per Thread (µs) - The average execution time per thread. Lower values indicate more efficient execution per thread.',
    'Compute/Memory Efficiency Ratio': 'Compute/Memory Efficiency Ratio = Compute throughput divided by memory throughput. Higher values indicate better compute-bound performance.',
    'Memory Efficiency': 'Memory Efficiency - Memory throughput divided by execution time. Higher values indicate more efficient memory access.',
    'Normalized Occupancy': 'Normalized Occupancy (%) - Achieved occupancy divided by theoretical occupancy. Provides a measure of how well the kernel is utilizing the available resources.',
    'Normalized SM Efficiency': 'Normalized SM Efficiency (%) - SM efficiency divided by theoretical occupancy. A measure of how well the SMs are utilized relative to the occupancy.',
    'Warp Execution Ratio': 'Warp Execution Ratio = Achieved warps divided by warp limit per block. Indicates how effectively warps are being utilized within the given block configuration.',
    'Memory Bandwidth Utilization': 'Memory Bandwidth Utilization - Memory throughput divided by L2 cache throughput. Higher values indicate more effective use of memory bandwidth.',
    'Compute Throughput vs Execution Time': 'Compute Throughput vs Execution Time - The ratio of compute throughput to execution time. Higher values indicate more efficient kernel performance.',
    'Dynamic Shared Memory Usage': 'Dynamic Shared Memory Usage (KB) - The dynamic shared memory allocated per block. Affects how much data can be stored on-chip for use by the threads.',
    'Achieved Active Warps': 'Achieved Active Warps - The actual number of active warps achieved compared to the theoretical maximum. Higher values indicate better warp utilization.',
    'Achieved Warps vs Occupancy': 'Achieved Warps vs Occupancy - Shows the relationship between the number of warps achieved and the occupancy of the kernel. Higher values indicate better utilization of the GPU resources.',
    'SM Active Cycles': 'SM Active Cycles - The number of cycles where the streaming multiprocessors are actively executing instructions. Higher values indicate more active use of the compute units.',
    'Cache Throughput vs SM Active Cycles': 'Cache Throughput vs SM Active Cycles - Shows the relationship between cache throughput and SM activity. Higher values indicate better use of on-chip resources.',
    'Register Usage Efficiency': 'Register Usage Efficiency - Registers per thread divided by the register limit per block. Higher values indicate better utilization of register resources.',
    'Shared Memory Usage Efficiency': 'Shared Memory Usage Efficiency - Shared memory usage per block divided by the maximum shared memory available. Higher values indicate better utilization of shared memory resources.',
    'Occupancy vs Performance': 'Occupancy vs Performance - Shows the balance between occupancy and execution time. Ideal performance is indicated by high occupancy and low execution time.',
    'Achieved Warps vs Occupancy': 'Achieved Warps vs Occupancy - Shows the relationship between the number of warps achieved and the occupancy of the kernel. Higher values indicate better utilization of the GPU resources.',
    'Achieved Occupancy vs Duration': 'Achieved Occupancy vs Duration - Shows how achieved occupancy correlates with execution time. Higher occupancy with lower execution time is ideal.',
}






# KPI calculations including new derived metrics
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
    
    # Additional KPIs for analysis
    df['Compute Throughput vs Execution Time'] = df['computeSMThroughputPercent'] / df['durationUsecond']
    df['Memory Bandwidth Utilization'] = df['memoryThroughputPercent'] / df['l2CacheThroughputPercent']
    df['Warp Execution Efficiency'] = df['achievedActiveWarpsPerSm'] / df['theoreticalActiveWarpsPerSm']
    df['Performance Variability'] = df.groupby('block_config')['durationUsecond'].transform(lambda x: x.std() / x.mean())
    
    # 5 New Derived Metrics
    df['SM Active Utilization'] = df['smActiveCycles'] / df['elapsedCycles']  # Utilization of SMs
    df['Warp Execution Ratio'] = df['achievedActiveWarpsPerSm'] / df['blockLimitWarps']  # Ratio of active warps to warp limit
    df['Memory Access Efficiency'] = df['memoryThroughputPercent'] / df['durationUsecond']  # Efficiency of memory access
    df['Register Usage Efficiency'] = df['registersPerThread'] / df['blockLimitRegisters']  # Efficiency of register usage
    df['Shared Memory Usage Efficiency'] = df['sharedMemoryConfigSizeKbyte'] / df['blockLimitSharedMem']  # Efficiency of shared memory usage
    
    return df

# Calculate KPIs with updated metrics
filtered_data = calculate_kpi(filtered_data)

# Function to explain each graph in detail
def explain_graph(title, x_axis, y_axis, derived_metrics=None):
    """
    This function displays detailed explanations and analyses for each graph.
    
    Parameters:
    - title: The title of the graph.
    - x_axis: The x-axis label and its description.
    - y_axis: The y-axis label and its description.
    - derived_metrics: Explanation of any derived metrics used in the graph (optional).
    """
    st.write(f"### {title}")
    st.write(f"**X-axis:** {x_axis} - This represents the {x_axis.lower()} value across different configurations. It is crucial for understanding how this variable affects the outcome of the graph.")
    st.write(f"**Y-axis:** {y_axis} - This represents the {y_axis.lower()} observed during the execution. It gives insight into how the kernel performs under different {x_axis.lower()} values.")
    
    if derived_metrics:
        st.write(f"**Derived Metrics:** {derived_metrics}")
    
    st.write(f"**Graph Analysis:** This graph shows the relationship between {x_axis.lower()} and {y_axis.lower()}. Look for patterns such as linear relationships, plateaus, or peaks, which can indicate performance bottlenecks or optimal configurations. ")
    st.write(f"For instance, a plateau might indicate a saturation point where increasing {x_axis.lower()} no longer improves {y_axis.lower()}, suggesting a bottleneck in the system. Sudden peaks can indicate an inefficient configuration or excessive resource contention.")


# Analysis options for KPIs - Updated with additional options
analysis_option = st.sidebar.selectbox("Choose Analysis", [
    'Execution Time vs Block Configuration',
    'SM Efficiency',
    'Compute to Memory Ratio',
    'Memory Throughput vs Block Configuration',
    'DRAM vs SM Frequency',
    'Cache Hit Rate Analysis',
    'Occupancy Analysis',
    'Warp Execution Efficiency',
    'Register Pressure Analysis',
    'Duration Useconds Distribution',
    'Elapsed Cycles Analysis',
    'L1 Cache vs L2 Cache Throughput',
    'SM Active Cycles vs Duration',
    'Achieved Occupancy vs Duration',
    'Compute SM Throughput vs Block Size',
    'DRAM Throughput vs Block Size',
    'SM Frequency Cycle vs Block Configuration',
    'Grid Size Impact on Performance',
    'Performance Variability',
    'Execution Time Distribution',
    'Block Limit Warps vs Performance',
    'Block Limit Shared Memory vs Registers',
    'Achieved Warps vs Occupancy',
    'SM Utilization vs Memory Throughput',
    'Compute Throughput vs DRAM Throughput',
    'Cache Throughput vs SM Active Cycles',
    'Memory Bandwidth Utilization',
    'Compute Throughput vs Execution Time',  # Added missing KPI
    'SM Active Utilization',  # New KPI
    'Warp Execution Ratio',  # New KPI
    'Memory Access Efficiency',  # New KPI
    'Register Usage Efficiency',  # New KPI
    'Shared Memory Usage Efficiency',  # New KPI
    'Occupancy vs Performance 3D Plot',
    'Register Pressure vs Performance',
])

# Main content: Implement all the analyses based on the selected option
st.write(f"### Analysis: {analysis_option}")

# Logic to generate selected analysis graphs
if analysis_option == 'Execution Time vs Block Configuration':
    explain_graph(
        title='Execution Time vs Block Configuration', 
        x_axis='block_config', 
        y_axis='durationUsecond'
    )
    scatter_plot('block_config', 'durationUsecond', size='durationUsecond', title='Execution Time vs Block Configuration', 
                 x_label='Block Configuration', y_label='Execution Time (µs)', size_label='Execution Time (µs)')

elif analysis_option == 'SM Efficiency':
    explain_graph(
        title='SM Efficiency vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='SM Efficiency', 
        derived_metrics='SM Efficiency = compute throughput / (SM frequency * elapsed cycles)'
    )
    scatter_plot('block_config', 'SM Efficiency', size='SM Efficiency', title='SM Efficiency vs Block Configuration',
                 x_label='Block Configuration', y_label='SM Efficiency', size_label='Efficiency')

elif analysis_option == 'Compute to Memory Ratio':
    explain_graph(
        title='Compute to Memory Ratio vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Compute to Memory Ratio', 
        derived_metrics='Compute to Memory Ratio = SM active cycles / elapsed cycles'
    )
    scatter_plot('block_config', 'Compute to Memory Ratio', size='Compute to Memory Ratio', title='Compute to Memory Ratio vs Block Configuration',
                 x_label='Block Configuration', y_label='Compute to Memory Ratio', size_label='Ratio')

elif analysis_option == 'Memory Throughput vs Block Configuration':
    explain_graph(
        title='Memory Throughput vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Memory Throughput (%)', 
        derived_metrics='Memory Throughput = Total memory used by the kernel / available bandwidth'
    )
    scatter_plot('block_config', 'memoryThroughputPercent', size='memoryThroughputPercent', title='Memory Throughput vs Block Configuration',
                 x_label='Block Configuration', y_label='Memory Throughput (%)', size_label='Memory Throughput (%)')

elif analysis_option == 'DRAM vs SM Frequency':
    explain_graph(
        title='DRAM vs SM Frequency vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='DRAM vs SM Frequency', 
        derived_metrics='DRAM vs SM Frequency = DRAM throughput / SM frequency'
    )
    scatter_plot('block_config', 'DRAM vs SM Frequency', size='DRAM vs SM Frequency', title='DRAM vs SM Frequency vs Block Configuration',
                 x_label='Block Configuration', y_label='DRAM vs SM Frequency', size_label='DRAM vs SM Frequency')

elif analysis_option == 'Cache Hit Rate Analysis':
    explain_graph(
        title='Cache Hit Rate vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Cache Hit Rate (%)', 
        derived_metrics='Cache Hit Rate = (L1 + L2 Cache Throughput) / Memory Throughput'
    )
    scatter_plot('block_config', 'Cache Hit Rate', size='Cache Hit Rate', title='Cache Hit Rate vs Block Configuration',
                 x_label='Block Configuration', y_label='Cache Hit Rate', size_label='Cache Hit Rate')

elif analysis_option == 'Occupancy Analysis':
    explain_graph(
        title='Occupancy vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Occupancy (%)', 
        derived_metrics='Occupancy = Achieved warps / Theoretical warps'
    )
    scatter_plot('block_config', 'Occupancy', size='Occupancy', title='Occupancy vs Block Configuration',
                 x_label='Block Configuration', y_label='Occupancy (%)', size_label='Occupancy (%)')

elif analysis_option == 'Warp Execution Efficiency':
    explain_graph(
        title='Warp Execution Efficiency vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Warp Execution Efficiency', 
        derived_metrics='Warp Execution Efficiency = Achieved active warps / Theoretical active warps'
    )
    scatter_plot('block_config', 'Warp Execution Efficiency', size='Warp Execution Efficiency', title='Warp Execution Efficiency',
                 x_label='Block Configuration', y_label='Warp Execution Efficiency', size_label='Warp Execution Efficiency')

elif analysis_option == 'Compute Throughput vs Execution Time':
    explain_graph(
        title='Compute Throughput vs Execution Time', 
        x_axis='Compute Throughput (%)', 
        y_axis='Execution Time (µs)', 
        derived_metrics='Compute Throughput vs Execution Time = Compute throughput / Execution time'
    )
    scatter_plot('computeSMThroughputPercent', 'durationUsecond', size='computeSMThroughputPercent', title='Compute Throughput vs Execution Time',
                 x_label='Compute Throughput (%)', y_label='Execution Time (µs)', size_label='Compute Throughput (%)')

elif analysis_option == 'Register Pressure Analysis':
    explain_graph(
        title='Register Pressure Analysis', 
        x_axis='Registers Per Thread', 
        y_axis='Execution Time (µs)', 
        derived_metrics='Register Pressure = Registers used by each thread. High values can decrease performance due to lower occupancy.'
    )
    scatter_plot('registersPerThread', 'durationUsecond', size='registersPerThread', title='Registers Per Thread vs Execution Time',
                 x_label='Registers Per Thread', y_label='Execution Time (µs)', size_label='Registers Per Thread')

elif analysis_option == 'Duration Useconds Distribution':
    explain_graph(
        title='Execution Time Distribution', 
        x_axis='Execution Time (µs)', 
        y_axis='Frequency', 
        derived_metrics='Duration Useconds Distribution = Spread of kernel execution times, helps in identifying performance outliers.'
    )
    histogram('durationUsecond', title='Execution Time Distribution in Microseconds', x_label='Execution Time (µs)')

elif analysis_option == 'Elapsed Cycles Analysis':
    explain_graph(
        title='Elapsed Cycles Analysis', 
        x_axis='Block Configuration', 
        y_axis='Elapsed Cycles', 
        derived_metrics='Elapsed Cycles = Total cycles taken for the kernel execution. Lower values indicate faster execution.'
    )
    scatter_plot('block_config', 'elapsedCycles', size='elapsedCycles', title='Elapsed Cycles vs Block Configuration',
                 x_label='Block Configuration', y_label='Elapsed Cycles', size_label='Elapsed Cycles')

elif analysis_option == 'L1 Cache vs L2 Cache Throughput':
    explain_graph(
        title='L1 vs L2 Cache Throughput', 
        x_axis='L1 Cache Throughput (%)', 
        y_axis='L2 Cache Throughput (%)', 
        derived_metrics='L1 vs L2 Cache Throughput = Utilization of L1 and L2 caches. Higher values indicate better cache usage.'
    )
    scatter_plot('l1TexCacheThroughputPercent', 'l2CacheThroughputPercent', title='L1 vs L2 Cache Throughput',
                 x_label='L1 Cache Throughput (%)', y_label='L2 Cache Throughput (%)')

elif analysis_option == 'SM Active Cycles vs Duration':
    explain_graph(
        title='SM Active Cycles vs Duration', 
        x_axis='SM Active Cycles', 
        y_axis='Execution Time (µs)', 
        derived_metrics='SM Active Cycles = Number of cycles where SMs are actively processing. High values with low execution time indicate efficient execution.'
    )
    scatter_plot('smActiveCycles', 'durationUsecond', size='durationUsecond', title='SM Active Cycles vs Execution Time',
                 x_label='SM Active Cycles', y_label='Execution Time (µs)', size_label='Execution Time (µs)')

elif analysis_option == 'Achieved Occupancy vs Duration':
    explain_graph(
        title='Achieved Occupancy vs Duration', 
        x_axis='Achieved Occupancy (%)', 
        y_axis='Execution Time (µs)', 
        derived_metrics='Achieved Occupancy = Actual occupancy relative to theoretical occupancy. Higher values indicate better GPU utilization.'
    )
    scatter_plot('achievedOccupancyPercent', 'durationUsecond', size='durationUsecond', title='Achieved Occupancy vs Execution Time',
                 x_label='Achieved Occupancy (%)', y_label='Execution Time (µs)', size_label='Execution Time (µs)')

elif analysis_option == 'Compute SM Throughput vs Block Size':
    explain_graph(
        title='Compute SM Throughput vs Block Size', 
        x_axis='Block Configuration', 
        y_axis='Compute SM Throughput (%)', 
        derived_metrics='Compute SM Throughput = Efficiency of the streaming multiprocessors. Higher values indicate better compute utilization.'
    )
    scatter_plot('block_config', 'computeSMThroughputPercent', size='computeSMThroughputPercent', title='Compute SM Throughput vs Block Size',
                 x_label='Block Configuration', y_label='Compute SM Throughput (%)', size_label='Compute SM Throughput (%)')

elif analysis_option == 'DRAM Throughput vs Block Size':
    explain_graph(
        title='DRAM Throughput vs Block Size', 
        x_axis='Block Configuration', 
        y_axis='DRAM Throughput (%)', 
        derived_metrics='DRAM Throughput = Memory bandwidth utilization. Higher values indicate efficient memory usage.'
    )
    scatter_plot('block_config', 'dramThroughputPercent', size='dramThroughputPercent', title='DRAM Throughput vs Block Size',
                 x_label='Block Configuration', y_label='DRAM Throughput (%)', size_label='DRAM Throughput (%)')

elif analysis_option == 'SM Frequency Cycle vs Block Configuration':
    explain_graph(
        title='SM Frequency Cycle vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='SM Frequency Cycle (cycles/usecond)', 
        derived_metrics='SM Frequency Cycle = Number of SM cycles per microsecond. Higher values indicate faster compute performance.'
    )
    scatter_plot('block_config', 'smFrequencyCyclePerUsecond', title='SM Frequency Cycle vs Block Configuration',
                 x_label='Block Configuration', y_label='SM Frequency Cycle (cycles/usecond)')

# Continue from where the previous logic left off

elif analysis_option == 'Grid Size Impact on Performance':
    explain_graph(
        title='Grid Size Impact on Performance', 
        x_axis='Grid Size', 
        y_axis='Execution Time (µs)', 
        derived_metrics='Grid Size = Total number of blocks per dimension. Larger grid sizes can increase execution time due to overhead.'
    )
    scatter_plot('gridSize', 'durationUsecond', size='gridSize', title='Grid Size Impact on Execution Time',
                 x_label='Grid Size', y_label='Execution Time (µs)', size_label='Grid Size')

elif analysis_option == 'Performance Variability':
    explain_graph(
        title='Performance Variability Across Block Configurations', 
        x_axis='Block Configuration', 
        y_axis='Performance Variability', 
        derived_metrics='Performance Variability = Standard deviation of execution time divided by mean execution time. Lower values indicate consistent performance.'
    )
    scatter_plot('block_config', 'Performance Variability', title='Performance Variability Across Block Configurations',
                 x_label='Block Configuration', y_label='Performance Variability')

elif analysis_option == 'Execution Time Distribution':
    explain_graph(
        title='Execution Time Distribution', 
        x_axis='Execution Time (µs)', 
        y_axis='Frequency', 
        derived_metrics='Execution Time Distribution = Histogram showing the distribution of execution times.'
    )
    histogram('durationUsecond', title='Execution Time Distribution', x_label='Execution Time (µs)')

elif analysis_option == 'Block Limit Warps vs Performance':
    explain_graph(
        title='Block Limit Warps vs Execution Time', 
        x_axis='Block Limit Warps', 
        y_axis='Execution Time (µs)', 
        derived_metrics='Block Limit Warps = Number of warps allowed per block. More warps per block generally improve performance up to a certain limit.'
    )
    scatter_plot('blockLimitWarps', 'durationUsecond', size='blockLimitWarps', title='Block Limit Warps vs Execution Time',
                 x_label='Block Limit Warps', y_label='Execution Time (µs)', size_label='Block Limit Warps')

elif analysis_option == 'Block Limit Shared Memory vs Registers':
    explain_graph(
        title='Block Limit Shared Memory vs Registers Per Thread', 
        x_axis='Block Limit Shared Memory', 
        y_axis='Registers Per Thread', 
        derived_metrics='Block Limit = Maximum shared memory and register usage. Balancing both is crucial for optimal performance.'
    )
    scatter_plot('blockLimitSharedMem', 'registersPerThread', size='blockLimitSharedMem', title='Block Limit Shared Memory vs Registers Per Thread',
                 x_label='Block Limit Shared Memory', y_label='Registers Per Thread', size_label='Block Limit Shared Memory')

elif analysis_option == 'Achieved Warps vs Occupancy':
    explain_graph(
        title='Achieved Warps vs Occupancy', 
        x_axis='Achieved Warps per SM', 
        y_axis='Achieved Occupancy (%)', 
        derived_metrics='Achieved Warps = Number of warps actively utilized per SM. Higher values indicate better GPU resource utilization.'
    )
    scatter_plot('achievedActiveWarpsPerSm', 'achievedOccupancyPercent', size='achievedActiveWarpsPerSm', title='Achieved Warps vs Occupancy',
                 x_label='Achieved Warps per SM', y_label='Achieved Occupancy (%)', size_label='Achieved Warps per SM')

elif analysis_option == 'SM Utilization vs Memory Throughput':
    explain_graph(
        title='SM Utilization vs Memory Throughput', 
        x_axis='SM Active Cycles', 
        y_axis='Memory Throughput (%)', 
        derived_metrics='SM Utilization = How efficiently the streaming multiprocessors are used. Higher utilization with good memory throughput indicates balanced performance.'
    )
    scatter_plot('smActiveCycles', 'memoryThroughputPercent', size='smActiveCycles', title='SM Utilization vs Memory Throughput',
                 x_label='SM Active Cycles', y_label='Memory Throughput (%)', size_label='SM Active Cycles')

elif analysis_option == 'Compute Throughput vs DRAM Throughput':
    explain_graph(
        title='Compute Throughput vs DRAM Throughput', 
        x_axis='Compute SM Throughput (%)', 
        y_axis='DRAM Throughput (%)', 
        derived_metrics='Compute vs DRAM Throughput = Balance between computation and memory bandwidth. Ideally, both should be high for efficient execution.'
    )
    scatter_plot('computeSMThroughputPercent', 'dramThroughputPercent', size='durationUsecond', title='Compute Throughput vs DRAM Throughput',
                 x_label='Compute Throughput (%)', y_label='DRAM Throughput (%)', size_label='Execution Time (µs)')

elif analysis_option == 'Cache Throughput vs SM Active Cycles':
    explain_graph(
        title='Cache Throughput vs SM Active Cycles', 
        x_axis='L1 Cache Throughput (%)', 
        y_axis='SM Active Cycles', 
        derived_metrics='Cache Throughput = Higher cache throughput reduces DRAM accesses, improving performance.'
    )
    scatter_plot('l1TexCacheThroughputPercent', 'smActiveCycles', size='durationUsecond', title='Cache Throughput vs SM Active Cycles',
                 x_label='L1 Cache Throughput (%)', y_label='SM Active Cycles', size_label='Execution Time (µs)')

elif analysis_option == 'Memory Bandwidth Utilization':
    explain_graph(
        title='Memory Bandwidth Utilization vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Memory Bandwidth Utilization', 
        derived_metrics='Memory Bandwidth Utilization = Memory throughput divided by available bandwidth. Higher values indicate better utilization of available bandwidth.'
    )
    scatter_plot('block_config', 'Memory Bandwidth Utilization', title='Memory Bandwidth Utilization vs Block Configuration',
                 x_label='Block Configuration', y_label='Memory Bandwidth Utilization')

elif analysis_option == 'SM Active Utilization':
    explain_graph(
        title='SM Active Utilization vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='SM Active Utilization', 
        derived_metrics='SM Active Utilization = Active cycles of SMs divided by total cycles. Higher values indicate better utilization of streaming multiprocessors.'
    )
    scatter_plot('block_config', 'SM Active Utilization', size='SM Active Utilization', title='SM Active Utilization vs Block Configuration',
                 x_label='Block Configuration', y_label='SM Active Utilization', size_label='SM Active Utilization')

elif analysis_option == 'Warp Execution Ratio':
    explain_graph(
        title='Warp Execution Ratio vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Warp Execution Ratio', 
        derived_metrics='Warp Execution Ratio = Achieved warps divided by warp limit per block. Indicates how effectively warps are utilized.'
    )
    scatter_plot('block_config', 'Warp Execution Ratio', size='Warp Execution Ratio', title='Warp Execution Ratio vs Block Configuration',
                 x_label='Block Configuration', y_label='Warp Execution Ratio', size_label='Warp Execution Ratio')

elif analysis_option == 'Memory Access Efficiency':
    explain_graph(
        title='Memory Access Efficiency vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Memory Access Efficiency', 
        derived_metrics='Memory Access Efficiency = Memory throughput divided by execution time. Higher values indicate more efficient memory access.'
    )
    scatter_plot('block_config', 'Memory Access Efficiency', size='Memory Access Efficiency', title='Memory Access Efficiency vs Block Configuration',
                 x_label='Block Configuration', y_label='Memory Access Efficiency', size_label='Memory Access Efficiency')

elif analysis_option == 'Register Usage Efficiency':
    explain_graph(
        title='Register Usage Efficiency vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Register Usage Efficiency', 
        derived_metrics='Register Usage Efficiency = Registers per thread divided by register limit per block. Higher values indicate more efficient register usage.'
    )
    scatter_plot('block_config', 'Register Usage Efficiency', size='Register Usage Efficiency', title='Register Usage Efficiency vs Block Configuration',
                 x_label='Block Configuration', y_label='Register Usage Efficiency', size_label='Register Usage Efficiency')

elif analysis_option == 'Shared Memory Usage Efficiency':
    explain_graph(
        title='Shared Memory Usage Efficiency vs Block Configuration', 
        x_axis='Block Configuration', 
        y_axis='Shared Memory Usage Efficiency', 
        derived_metrics='Shared Memory Usage Efficiency = Shared memory usage per block divided by block limit for shared memory. Higher values indicate better utilization of shared memory.'
    )
    scatter_plot('block_config', 'Shared Memory Usage Efficiency', size='Shared Memory Usage Efficiency', title='Shared Memory Usage Efficiency vs Block Configuration',
                 x_label='Block Configuration', y_label='Shared Memory Usage Efficiency', size_label='Shared Memory Usage Efficiency')

elif analysis_option == 'Occupancy vs Performance 3D Plot':
    explain_graph(
        title='Occupancy vs Compute Throughput vs Execution Time', 
        x_axis='Achieved Occupancy (%)', 
        y_axis='Compute SM Throughput (%)', 
        derived_metrics='Occupancy vs Performance = Shows the balance between occupancy, compute throughput, and execution time. Ideal performance is indicated by high occupancy and throughput with low execution time.'
    )
    scatter_3d('achievedOccupancyPercent', 'computeSMThroughputPercent', 'durationUsecond', title='Occupancy vs Compute Throughput vs Execution Time',
               x_label='Achieved Occupancy (%)', y_label='Compute SM Throughput (%)', z_label='Execution Time (µs)')

elif analysis_option == 'Register Pressure vs Performance':
    explain_graph(
        title='Register Pressure vs Achieved Warps', 
        x_axis='Registers Per Thread', 
        y_axis='Achieved Warps per SM', 
        derived_metrics='Register Pressure = Impact of register usage on warp execution efficiency. Higher register pressure can decrease performance due to reduced occupancy.'
    )
    scatter_plot('registersPerThread', 'achievedActiveWarpsPerSm', size='durationUsecond', title='Register Pressure vs Achieved Warps',
                 x_label='Registers Per Thread', y_label='Achieved Warps per SM', size_label='Execution Time (µs)')

elif analysis_option == 'Warp Occupancy vs Memory Throughput':
    explain_graph(
        title='Warp Occupancy vs Memory Throughput', 
        x_axis='Achieved Occupancy (%)', 
        y_axis='Memory Throughput (%)', 
        derived_metrics='Warp Occupancy = Ratio of active warps to theoretical warps. Higher values with good memory throughput indicate balanced GPU usage.'
    )
    scatter_plot('achievedOccupancyPercent', 'memoryThroughputPercent', size='durationUsecond', title='Warp Occupancy vs Memory Throughput',
                 x_label='Achieved Occupancy (%)', y_label='Memory Throughput (%)', size_label='Execution Time (µs)')

elif analysis_option == 'L2 Cache Hit Rate vs SM Efficiency':
    explain_graph(
        title='L2 Cache Hit Rate vs SM Efficiency', 
        x_axis='L2 Cache Throughput (%)', 
        y_axis='SM Efficiency', 
        derived_metrics='L2 Cache Hit Rate = Efficiency of accessing data from L2 cache. Higher values indicate more efficient memory access with less DRAM usage.'
    )
    scatter_plot('l2CacheThroughputPercent', 'SM Efficiency', size='durationUsecond', title='L2 Cache Hit Rate vs SM Efficiency',
                 x_label='L2 Cache Throughput (%)', y_label='SM Efficiency', size_label='Execution Time (µs)')

elif analysis_option == 'DRAM Read vs Write Throughput':
    explain_graph(
        title='DRAM Read vs Write Throughput', 
        x_axis='DRAM Read Throughput (%)', 
        y_axis='DRAM Write Throughput (%)', 
        derived_metrics='DRAM Read vs Write Throughput = Balance between reading and writing data to DRAM. Ideally, both should be high for balanced memory access.'
    )
    scatter_plot('dramReadThroughputPercent', 'dramWriteThroughputPercent', size='durationUsecond', title='DRAM Read vs Write Throughput',
                 x_label='DRAM Read Throughput (%)', y_label='DRAM Write Throughput (%)', size_label='Execution Time (µs)')

elif analysis_option == 'Theoretical vs Achieved Occupancy':
    explain_graph(
        title='Theoretical vs Achieved Occupancy', 
        x_axis='Theoretical Occupancy (%)', 
        y_axis='Achieved Occupancy (%)', 
        derived_metrics='Theoretical vs Achieved Occupancy = Comparison between theoretical and achieved occupancy. Ideally, both should be close to maximize GPU utilization.'
    )
    scatter_plot('theoreticalOccupancyPercent', 'achievedOccupancyPercent', size='durationUsecond', title='Theoretical vs Achieved Occupancy',
                 x_label='Theoretical Occupancy (%)', y_label='Achieved Occupancy (%)', size_label='Execution Time (µs)')

elif analysis_option == 'FMA Utilization vs SM Efficiency':
    explain_graph(
        title='FMA Utilization vs SM Efficiency', 
        x_axis='FMA Utilization (%)', 
        y_axis='SM Efficiency', 
        derived_metrics='FMA Utilization = Efficiency of fused multiply-add operations. Higher values with high SM efficiency indicate good compute performance.'
    )
    scatter_plot('fmaUtilizationPercent', 'SM Efficiency', size='durationUsecond', title='FMA Utilization vs SM Efficiency',
                 x_label='FMA Utilization (%)', y_label='SM Efficiency', size_label='Execution Time (µs)')

elif analysis_option == 'L1 Cache vs DRAM Throughput':
    explain_graph(
        title='L1 Cache vs DRAM Throughput', 
        x_axis='L1 Cache Throughput (%)', 
        y_axis='DRAM Throughput (%)', 
        derived_metrics='L1 Cache Throughput = Efficiency of accessing data from L1 cache. High values with balanced DRAM throughput indicate good memory access patterns.'
    )
    scatter_plot('l1TexCacheThroughputPercent', 'dramThroughputPercent', size='durationUsecond', title='L1 Cache vs DRAM Throughput',
                 x_label='L1 Cache Throughput (%)', y_label='DRAM Throughput (%)', size_label='Execution Time (µs)')

elif analysis_option == 'Achieved Warps vs Execution Time':
    explain_graph(
        title='Achieved Warps vs Execution Time', 
        x_axis='Achieved Active Warps per SM', 
        y_axis='Execution Time (µs)', 
        derived_metrics='Achieved Warps = Actual warps actively utilized per SM. Higher values with low execution time indicate good GPU utilization.'
    )
    scatter_plot('achievedActiveWarpsPerSm', 'durationUsecond', size='achievedActiveWarpsPerSm', title='Achieved Warps vs Execution Time',
                 x_label='Achieved Active Warps per SM', y_label='Execution Time (µs)', size_label='Achieved Warps per SM')

elif analysis_option == 'Cache Efficiency vs Memory Throughput':
    explain_graph(
        title='Cache Efficiency vs Memory Throughput', 
        x_axis='Cache Efficiency', 
        y_axis='Memory Throughput (%)', 
        derived_metrics='Cache Efficiency = Ratio of cache accesses to DRAM accesses. Higher values indicate better use of cache memory over DRAM.'
    )
    scatter_plot('cacheEfficiencyPercent', 'memoryThroughputPercent', size='durationUsecond', title='Cache Efficiency vs Memory Throughput',
                 x_label='Cache Efficiency', y_label='Memory Throughput (%)', size_label='Execution Time (µs)')

elif analysis_option == 'Shared Memory Efficiency vs Registers Per Thread':
    explain_graph(
        title='Shared Memory Efficiency vs Registers Per Thread', 
        x_axis='Shared Memory Efficiency', 
        y_axis='Registers Per Thread', 
        derived_metrics='Shared Memory Efficiency = Efficient use of shared memory relative to available space. Higher values indicate better utilization of shared memory with respect to registers used.'
    )
    scatter_plot('sharedMemoryEfficiencyPercent', 'registersPerThread', size='durationUsecond', title='Shared Memory Efficiency vs Registers Per Thread',
                 x_label='Shared Memory Efficiency', y_label='Registers Per Thread', size_label='Execution Time (µs)')

elif analysis_option == 'Warp Efficiency vs Compute Throughput':
    explain_graph(
        title='Warp Efficiency vs Compute Throughput', 
        x_axis='Warp Efficiency (%)', 
        y_axis='Compute Throughput (%)', 
        derived_metrics='Warp Efficiency = Efficiency of warp execution. Higher values with good compute throughput indicate balanced kernel performance.'
    )
    scatter_plot('warpEfficiencyPercent', 'computeSMThroughputPercent', size='durationUsecond', title='Warp Efficiency vs Compute Throughput',
                 x_label='Warp Efficiency (%)', y_label='Compute Throughput (%)', size_label='Execution Time (µs)')

st.sidebar.subheader('Add Additional Graphs')
new_x_axis = st.sidebar.selectbox('Choose X-axis', data.columns)
new_y_axis = st.sidebar.selectbox('Choose Y-axis', data.columns)
add_new_graph = st.sidebar.button('Generate Additional Graph')

if add_new_graph:
    st.header(f'{new_x_axis} vs {new_y_axis}')
    scatter_plot(new_x_axis, new_y_axis, title=f'{new_x_axis} vs {new_y_axis}',
                 x_label=new_x_axis, y_label=new_y_axis)