
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys

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

# Function to create interactive scatter plots using Plotly
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
    st.plotly_chart(fig)

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
    
    return df

filtered_data = calculate_kpi(filtered_data)

# Allow the user to select 1 to 10 custom graphs
custom_graph_count = st.sidebar.slider('Select number of custom graphs to generate', min_value=1, max_value=10, value=1)

# Sidebar to select custom axes and titles for each graph
custom_graph_settings = []
for i in range(custom_graph_count):
    st.sidebar.subheader(f'Custom Graph {i + 1}')
    x_axis = st.sidebar.selectbox(f'Select X-axis for Graph {i + 1}', data.columns, key=f"x_axis_{i}")
    y_axis = st.sidebar.selectbox(f'Select Y-axis for Graph {i + 1}', data.columns, key=f"y_axis_{i}")
    size_axis = st.sidebar.selectbox(f'Select Size axis (optional) for Graph {i + 1}', [None] + list(data.columns), key=f"size_axis_{i}")
    graph_title = st.sidebar.text_input(f'Graph Title {i + 1}', value=f'{x_axis} vs {y_axis}', key=f"title_{i}")
    custom_graph_settings.append({'x': x_axis, 'y': y_axis, 'size': size_axis, 'title': graph_title})

# Render the custom graphs dynamically
for i, graph in enumerate(custom_graph_settings):
    st.header(f"Custom Graph {i + 1}: {graph['title']}")
    scatter_plot(graph['x'], graph['y'], size=graph['size'], title=graph['title'], 
                 x_label=graph['x'], y_label=graph['y'], size_label=graph['size'])
                 
# Analysis options for KPIs - 30+ different graph options
analysis_option = st.sidebar.selectbox("Choose Analysis", [
    'Execution Time vs Block Configuration',
    'SM Efficiency',
    'Compute to Memory Ratio',
    'Memory Throughput vs Block Configuration',
    'DRAM vs SM Frequency',
    'Cache Hit Rate Analysis',
    'Occupancy Analysis',
    'Shared Memory Usage',
    'Register Pressure Analysis',
    'Duration Useconds Distribution',
    'Elapsed Cycles Analysis',
    'L1 Cache vs L2 Cache Throughput',
    'SM Active Cycles vs Duration',
    'Warp Execution Efficiency',
    'Achieved Occupancy vs Duration',
    'Compute SM Throughput vs Block Size',
    'DRAM Throughput vs Block Size',
    'SM Frequency Cycle vs Block Configuration',
    'Grid Size Impact on Performance',
    'Dynamic Shared Memory Usage',
    'Memory Bandwidth Utilization',
    'Occupancy vs Performance 3D Plot',
    'Register Pressure vs Performance',
    'Performance Variability',
    'Execution Time Distribution',
    'Block Limit Warps vs Performance',
    'Block Limit Shared Memory vs Registers',
    'Achieved Warps vs Occupancy',
    'SM Utilization vs Memory Throughput',
    'Compute Throughput vs DRAM Throughput',
    'Cache Throughput vs SM Active Cycles',
])

# Main content: Implement all the analyses based on the selected option
st.write(f"### Analysis: {analysis_option}")




if analysis_option == 'Execution Time vs Block Configuration':
    st.write("This graph shows how different block configurations affect execution time. Look for configurations with lower execution times for better performance.")
    scatter_plot('block_config', 'durationUsecond', size='durationUsecond', title='Execution Time vs Block Configuration', 
                 x_label='Block Configuration', y_label='Execution Time (µs)', size_label='Execution Time (µs)')

elif analysis_option == 'SM Efficiency':
    st.write("This graph represents SM Efficiency, calculated as compute throughput divided by elapsed cycles. Higher values indicate better efficiency.")
    scatter_plot('block_config', 'SM Efficiency', size='SM Efficiency', title='SM Efficiency vs Block Configuration',
                 x_label='Block Configuration', y_label='SM Efficiency', size_label='Efficiency')

elif analysis_option == 'Compute to Memory Ratio':
    st.write("This metric indicates the balance between computation and memory access. A higher ratio implies better compute-bound kernel performance.")
    scatter_plot('block_config', 'Compute to Memory Ratio', size='Compute to Memory Ratio', title='Compute to Memory Ratio vs Block Configuration',
                 x_label='Block Configuration', y_label='Compute to Memory Ratio', size_label='Ratio')

elif analysis_option == 'Memory Throughput vs Block Configuration':
    st.write("Higher memory throughput indicates better use of memory bandwidth. Aim for configurations with higher memory throughput.")
    scatter_plot('block_config', 'memoryThroughputPercent', size='memoryThroughputPercent', title='Memory Throughput vs Block Configuration',
                 x_label='Block Configuration', y_label='Memory Throughput (%)', size_label='Memory Throughput (%)')

elif analysis_option == 'DRAM vs SM Frequency':
    st.write("This graph compares DRAM throughput with SM frequency. Higher values indicate better alignment between memory and compute performance.")
    scatter_plot('block_config', 'DRAM vs SM Frequency', size='DRAM vs SM Frequency', title='DRAM vs SM Frequency vs Block Configuration',
                 x_label='Block Configuration', y_label='DRAM vs SM Frequency', size_label='DRAM vs SM Frequency')

elif analysis_option == 'Cache Hit Rate Analysis':
    st.write("Higher cache hit rates indicate better use of cache memory, reducing slower DRAM accesses.")
    scatter_plot('block_config', 'Cache Hit Rate', size='Cache Hit Rate', title='Cache Hit Rate vs Block Configuration',
                 x_label='Block Configuration', y_label='Cache Hit Rate', size_label='Cache Hit Rate')

elif analysis_option == 'Occupancy Analysis':
    st.write("This graph shows the achieved occupancy. Higher occupancy indicates better utilization of GPU resources.")
    scatter_plot('block_config', 'Occupancy', size='Occupancy', title='Occupancy vs Block Configuration',
                 x_label='Block Configuration', y_label='Occupancy (%)', size_label='Occupancy (%)')

elif analysis_option == 'Shared Memory Usage':
    st.write("This graph shows the shared memory usage across different block configurations. Optimal shared memory usage can improve kernel performance.")
    scatter_plot('block_config', 'sharedMemoryConfigSizeKbyte', size='sharedMemoryConfigSizeKbyte', title='Shared Memory Usage vs Block Configuration',
                 x_label='Block Configuration', y_label='Shared Memory (KB)', size_label='Shared Memory (KB)')

elif analysis_option == 'Register Pressure Analysis':
    st.write("This graph shows the relationship between registers per thread and execution time. Higher register pressure can reduce occupancy and increase execution time.")
    scatter_plot('registersPerThread', 'durationUsecond', size='registersPerThread', title='Registers Per Thread vs Execution Time',
                 x_label='Registers Per Thread', y_label='Execution Time (µs)', size_label='Registers Per Thread')

elif analysis_option == 'Duration Useconds Distribution':
    st.write("This histogram shows the distribution of execution times in microseconds. This can help identify performance outliers.")
    histogram('durationUsecond', title='Execution Time Distribution in Microseconds', x_label='Execution Time (µs)')

elif analysis_option == 'Elapsed Cycles Analysis':
    st.write("This graph shows the number of elapsed cycles. Lower values indicate faster kernel execution.")
    scatter_plot('block_config', 'elapsedCycles', size='elapsedCycles', title='Elapsed Cycles vs Block Configuration',
                 x_label='Block Configuration', y_label='Elapsed Cycles', size_label='Elapsed Cycles')

elif analysis_option == 'L1 Cache vs L2 Cache Throughput':
    st.write("This graph compares the throughput of L1 and L2 caches. Higher throughput indicates better cache utilization.")
    scatter_plot('l1TexCacheThroughputPercent', 'l2CacheThroughputPercent', title='L1 vs L2 Cache Throughput',
                 x_label='L1 Cache Throughput (%)', y_label='L2 Cache Throughput (%)')

elif analysis_option == 'SM Active Cycles vs Duration':
    st.write("This graph shows how SM active cycles relate to execution time. Higher active cycles with lower duration indicate efficient execution.")
    scatter_plot('smActiveCycles', 'durationUsecond', size='durationUsecond', title='SM Active Cycles vs Execution Time',
                 x_label='SM Active Cycles', y_label='Execution Time (µs)', size_label='Execution Time (µs)')

elif analysis_option == 'Warp Execution Efficiency':
    st.write("This graph shows the efficiency of warp execution. Higher values indicate better warp utilization.")
    scatter_plot('block_config', 'achievedActiveWarpsPerSm', size='achievedActiveWarpsPerSm', title='Warp Execution Efficiency',
                 x_label='Block Configuration', y_label='Achieved Warps per SM', size_label='Achieved Warps per SM')

elif analysis_option == 'Achieved Occupancy vs Duration':
    st.write("This graph shows how achieved occupancy correlates with execution time. Higher occupancy with lower execution time is ideal.")
    scatter_plot('achievedOccupancyPercent', 'durationUsecond', size='durationUsecond', title='Achieved Occupancy vs Execution Time',
                 x_label='Achieved Occupancy (%)', y_label='Execution Time (µs)', size_label='Execution Time (µs)')

elif analysis_option == 'Compute SM Throughput vs Block Size':
    st.write("This graph shows the compute SM throughput for different block sizes. Higher throughput indicates more efficient compute utilization.")
    scatter_plot('block_config', 'computeSMThroughputPercent', size='computeSMThroughputPercent', title='Compute SM Throughput vs Block Size',
                 x_label='Block Configuration', y_label='Compute SM Throughput (%)', size_label='Compute SM Throughput (%)')

elif analysis_option == 'DRAM Throughput vs Block Size':
    st.write("This graph shows how DRAM throughput varies with different block sizes. Higher throughput means better memory bandwidth utilization.")
    scatter_plot('block_config', 'dramThroughputPercent', size='dramThroughputPercent', title='DRAM Throughput vs Block Size',
                 x_label='Block Configuration', y_label='DRAM Throughput (%)', size_label='DRAM Throughput (%)')

elif analysis_option == 'SM Frequency Cycle vs Block Configuration':
    st.write("This graph shows the SM frequency cycle for different block configurations. Higher SM frequency cycle means faster compute performance.")
    scatter_plot('block_config', 'smFrequencyCyclePerUsecond', title='SM Frequency Cycle vs Block Configuration',
                 x_label='Block Configuration', y_label='SM Frequency Cycle (cycles/usecond)')

elif analysis_option == 'Grid Size Impact on Performance':
    st.write("This graph shows how grid size impacts performance, measured by execution time. Higher grid size could increase execution time depending on configuration.")
    scatter_plot('gridSize', 'durationUsecond', size='gridSize', title='Grid Size Impact on Execution Time',
                 x_label='Grid Size', y_label='Execution Time (µs)', size_label='Grid Size')

elif analysis_option == 'Dynamic Shared Memory Usage':
    st.write("This graph shows dynamic shared memory usage across different block configurations. Optimal usage of shared memory can improve kernel performance.")
    scatter_plot('block_config', 'dynamicSharedMemoryPerBlockKbyte', size='dynamicSharedMemoryPerBlockKbyte', title='Dynamic Shared Memory Usage',
                 x_label='Block Configuration', y_label='Dynamic Shared Memory (KB)', size_label='Dynamic Shared Memory (KB)')

elif analysis_option == 'Memory Bandwidth Utilization':
    st.write("This graph shows the memory bandwidth utilization across block configurations. Higher utilization is better for memory-bound kernels.")
    filtered_data['Memory Bandwidth Utilization'] = filtered_data['memoryThroughputPercent'] / filtered_data['l2CacheThroughputPercent']
    scatter_plot('block_config', 'Memory Bandwidth Utilization', title='Memory Bandwidth Utilization vs Block Configuration',
                 x_label='Block Configuration', y_label='Memory Bandwidth Utilization')

elif analysis_option == 'Occupancy vs Performance 3D Plot':
    st.write("This 3D plot shows the relationship between occupancy, compute throughput, and execution time. Optimal performance is indicated by higher occupancy and throughput with lower execution time.")
    scatter_3d('achievedOccupancyPercent', 'computeSMThroughputPercent', 'durationUsecond', title='Occupancy vs Compute Throughput vs Execution Time',
               x_label='Achieved Occupancy (%)', y_label='Compute SM Throughput (%)', z_label='Execution Time (µs)')

elif analysis_option == 'Register Pressure vs Performance':
    st.write("This graph shows how register pressure (registers per thread) impacts warp execution efficiency and performance. Higher register pressure may reduce performance.")
    scatter_plot('registersPerThread', 'achievedActiveWarpsPerSm', size='durationUsecond', title='Register Pressure vs Achieved Warps',
                 x_label='Registers Per Thread', y_label='Achieved Warps per SM', size_label='Execution Time (µs)')

elif analysis_option == 'Performance Variability':
    st.write("This graph shows the variability in execution time across different block configurations. Lower variability is better for consistent performance.")
    filtered_data['Performance Variability'] = filtered_data.groupby('block_config')['durationUsecond'].transform(lambda x: x.std() / x.mean())
    scatter_plot('block_config', 'Performance Variability', title='Performance Variability Across Block Configurations',
                 x_label='Block Configuration', y_label='Performance Variability')

elif analysis_option == 'Execution Time Distribution':
    st.write("This histogram shows the distribution of execution times across different runs. Identifying clusters or outliers can help optimize kernel execution.")
    histogram('durationUsecond', title='Execution Time Distribution', x_label='Execution Time (µs)')

elif analysis_option == 'Block Limit Warps vs Performance':
    st.write("This graph shows how the number of warps per block impacts performance. Higher warps per block generally improve performance up to a limit.")
    scatter_plot('blockLimitWarps', 'durationUsecond', size='blockLimitWarps', title='Block Limit Warps vs Execution Time',
                 x_label='Block Limit Warps', y_label='Execution Time (µs)', size_label='Block Limit Warps')

elif analysis_option == 'Block Limit Shared Memory vs Registers':
    st.write("This graph shows the relationship between shared memory and registers per thread. It helps to balance both resources to maximize occupancy and performance.")
    scatter_plot('blockLimitSharedMem', 'registersPerThread', size='blockLimitSharedMem', title='Block Limit Shared Memory vs Registers Per Thread',
                 x_label='Block Limit Shared Memory', y_label='Registers Per Thread', size_label='Block Limit Shared Memory')

elif analysis_option == 'Achieved Warps vs Occupancy':
    st.write("This graph shows the relationship between achieved warps and occupancy. Higher warps and occupancy often result in better GPU utilization.")
    scatter_plot('achievedActiveWarpsPerSm', 'achievedOccupancyPercent', size='achievedActiveWarpsPerSm', title='Achieved Warps vs Occupancy',
                 x_label='Achieved Warps per SM', y_label='Achieved Occupancy (%)', size_label='Achieved Warps per SM')

elif analysis_option == 'SM Utilization vs Memory Throughput':
    st.write("This graph shows the relationship between SM utilization (active cycles) and memory throughput. Balancing both is crucial for optimal performance.")
    scatter_plot('smActiveCycles', 'memoryThroughputPercent', size='smActiveCycles', title='SM Utilization vs Memory Throughput',
                 x_label='SM Active Cycles', y_label='Memory Throughput (%)', size_label='SM Active Cycles')

elif analysis_option == 'Compute Throughput vs DRAM Throughput':
    st.write("This graph shows the relationship between compute throughput and DRAM throughput. Ideally, both should be balanced for efficient execution.")
    scatter_plot('computeSMThroughputPercent', 'dramThroughputPercent', size='durationUsecond', title='Compute Throughput vs DRAM Throughput',
                 x_label='Compute Throughput (%)', y_label='DRAM Throughput (%)', size_label='Execution Time (µs)')

elif analysis_option == 'Cache Throughput vs SM Active Cycles':
    st.write("This graph shows the relationship between cache throughput and SM active cycles. Efficient cache usage can reduce SM idle time and improve performance.")
    scatter_plot('l1TexCacheThroughputPercent', 'smActiveCycles', size='durationUsecond', title='Cache Throughput vs SM Active Cycles',
                 x_label='L1 Cache Throughput (%)', y_label='SM Active Cycles', size_label='Execution Time (µs)')

st.sidebar.subheader('Add Additional Graphs')
new_x_axis = st.sidebar.selectbox('Choose X-axis', data.columns)
new_y_axis = st.sidebar.selectbox('Choose Y-axis', data.columns)
add_new_graph = st.sidebar.button('Generate Additional Graph')

if add_new_graph:
    st.header(f'{new_x_axis} vs {new_y_axis}')
    scatter_plot(new_x_axis, new_y_axis, title=f'{new_x_axis} vs {new_y_axis}',
                 x_label=new_x_axis, y_label=new_y_axis)