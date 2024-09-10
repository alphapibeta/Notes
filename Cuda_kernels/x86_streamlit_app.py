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
# Function to create interactive scatter plots using Plotly
def scatter_plot(x, y, size=None, title=None):
    if size:
        # Handle NaN values in the size column by filling them with a default value (e.g., the median size)
        filtered_data[size] = filtered_data[size].fillna(filtered_data[size].median())
        fig = px.scatter(filtered_data, x=x, y=y, size=size, hover_name='block_config', title=title)
    else:
        fig = px.scatter(filtered_data, x=x, y=y, hover_name='block_config', title=title)
    st.plotly_chart(fig)


# Function to create interactive histograms using Plotly
def histogram(column, title=None):
    fig = px.histogram(filtered_data, x=column, nbins=50, title=title)
    st.plotly_chart(fig)

# Function to create interactive 3D scatter plots using Plotly
def scatter_3d(x, y, z, title=None):
    fig = px.scatter_3d(filtered_data, x=x, y=y, z=z, color='block_config', title=title)
    st.plotly_chart(fig)

# KPI calculations
def calculate_kpi(df):
    df['SM Efficiency'] = df['computeSMThroughputPercent'] / (df['smFrequencyCyclePerUsecond'] * df['elapsedCycles'])
    df['Compute to Memory Ratio'] = df['smActiveCycles'] / df['elapsedCycles']
    df['DRAM vs SM Frequency'] = df['dramThroughputPercent'] / df['smFrequencyCyclePerUsecond']
    df['Cache Hit Rate'] = (df['l1TexCacheThroughputPercent'] + df['l2CacheThroughputPercent']) / df['memoryThroughputPercent']
    df['Occupancy'] = df['achievedActiveWarpsPerSm'] / df['theoreticalActiveWarpsPerSm']
    return df

filtered_data = calculate_kpi(filtered_data)

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

if analysis_option == 'Execution Time vs Block Configuration':
    st.header('Execution Time vs Block Configuration')
    scatter_plot('block_config', 'durationUsecond', size='durationUsecond', title='Execution Time vs Block Configuration')

elif analysis_option == 'SM Efficiency':
    st.header('SM Efficiency')
    scatter_plot('block_config', 'SM Efficiency', size='SM Efficiency', title='SM Efficiency vs Block Configuration')

elif analysis_option == 'Compute to Memory Ratio':
    st.header('Compute to Memory Ratio')
    scatter_plot('block_config', 'Compute to Memory Ratio', size='Compute to Memory Ratio', title='Compute to Memory Ratio vs Block Configuration')

elif analysis_option == 'Memory Throughput vs Block Configuration':
    st.header('Memory Throughput vs Block Configuration')
    scatter_plot('block_config', 'memoryThroughputPercent', size='memoryThroughputPercent', title='Memory Throughput vs Block Configuration')

elif analysis_option == 'DRAM vs SM Frequency':
    st.header('DRAM vs SM Frequency')
    scatter_plot('block_config', 'DRAM vs SM Frequency', size='DRAM vs SM Frequency', title='DRAM vs SM Frequency vs Block Configuration')

elif analysis_option == 'Cache Hit Rate Analysis':
    st.header('Cache Hit Rate Analysis')
    scatter_plot('block_config', 'Cache Hit Rate', size='Cache Hit Rate', title='Cache Hit Rate vs Block Configuration')

elif analysis_option == 'Occupancy Analysis':
    st.header('Occupancy Analysis')
    scatter_plot('block_config', 'Occupancy', size='Occupancy', title='Occupancy vs Block Configuration')

elif analysis_option == 'Shared Memory Usage':
    st.header('Shared Memory Usage')
    scatter_plot('block_config', 'sharedMemoryConfigSizeKbyte', size='sharedMemoryConfigSizeKbyte', title='Shared Memory Usage vs Block Configuration')

elif analysis_option == 'Register Pressure Analysis':
    st.header('Register Pressure Analysis')
    scatter_plot('registersPerThread', 'durationUsecond', size='registersPerThread', title='Registers Per Thread vs Execution Time')

elif analysis_option == 'Duration Useconds Distribution':
    st.header('Duration Useconds Distribution')
    histogram('durationUsecond', title='Execution Time Distribution in Microseconds')

elif analysis_option == 'Elapsed Cycles Analysis':
    st.header('Elapsed Cycles Analysis')
    scatter_plot('block_config', 'elapsedCycles', size='elapsedCycles', title='Elapsed Cycles vs Block Configuration')

elif analysis_option == 'L1 Cache vs L2 Cache Throughput':
    st.header('L1 vs L2 Cache Throughput')
    scatter_plot('l1TexCacheThroughputPercent', 'l2CacheThroughputPercent', title='L1 vs L2 Cache Throughput')

elif analysis_option == 'SM Active Cycles vs Duration':
    st.header('SM Active Cycles vs Execution Time')
    scatter_plot('smActiveCycles', 'durationUsecond', size='durationUsecond', title='SM Active Cycles vs Execution Time')

elif analysis_option == 'Warp Execution Efficiency':
    st.header('Warp Execution Efficiency')
    scatter_plot('block_config', 'achievedActiveWarpsPerSm', size='achievedActiveWarpsPerSm', title='Warp Execution Efficiency')

elif analysis_option == 'Achieved Occupancy vs Duration':
    st.header('Achieved Occupancy vs Execution Time')
    scatter_plot('achievedOccupancyPercent', 'durationUsecond', size='durationUsecond', title='Achieved Occupancy vs Execution Time')

elif analysis_option == 'Compute SM Throughput vs Block Size':
    st.header('Compute SM Throughput vs Block Size')
    scatter_plot('block_config', 'computeSMThroughputPercent', size='computeSMThroughputPercent', title='Compute SM Throughput vs Block Size')

elif analysis_option == 'DRAM Throughput vs Block Size':
    st.header('DRAM Throughput vs Block Size')
    scatter_plot('block_config', 'dramThroughputPercent', size='dramThroughputPercent', title='DRAM Throughput vs Block Size')

elif analysis_option == 'SM Frequency Cycle vs Block Configuration':
    st.header('SM Frequency Cycle vs Block Configuration')
    scatter_plot('block_config', 'smFrequencyCyclePerUsecond', title='SM Frequency Cycle vs Block Configuration')

elif analysis_option == 'Grid Size Impact on Performance':
    st.header('Grid Size Impact on Performance')
    scatter_plot('gridSize', 'durationUsecond', size='gridSize', title='Grid Size Impact on Execution Time')

elif analysis_option == 'Dynamic Shared Memory Usage':
    st.header('Dynamic Shared Memory Usage')
    scatter_plot('block_config', 'dynamicSharedMemoryPerBlockKbyte', size='dynamicSharedMemoryPerBlockKbyte', title='Dynamic Shared Memory Usage')

elif analysis_option == 'Memory Bandwidth Utilization':
    st.header('Memory Bandwidth Utilization')
    filtered_data['Memory Bandwidth Utilization'] = filtered_data['memoryThroughputPercent'] / filtered_data['l2CacheThroughputPercent']
    scatter_plot('block_config', 'Memory Bandwidth Utilization', title='Memory Bandwidth Utilization vs Block Configuration')

elif analysis_option == 'Occupancy vs Performance 3D Plot':
    st.header('Occupancy vs Performance 3D Plot')
    scatter_3d('achievedOccupancyPercent', 'computeSMThroughputPercent', 'durationUsecond', title='Occupancy vs Compute Throughput vs Execution Time')

elif analysis_option == 'Register Pressure vs Performance':
    st.header('Register Pressure vs Performance')
    scatter_plot('registersPerThread', 'achievedActiveWarpsPerSm', size='durationUsecond', title='Register Pressure vs Achieved Warps')

elif analysis_option == 'Performance Variability':
    st.header('Performance Variability')
    filtered_data['Performance Variability'] = filtered_data.groupby('block_config')['durationUsecond'].transform(lambda x: x.std() / x.mean())
    scatter_plot('block_config', 'Performance Variability', title='Performance Variability Across Block Configurations')

elif analysis_option == 'Execution Time Distribution':
    st.header('Execution Time Distribution')
    histogram('durationUsecond', title='Execution Time Distribution')

elif analysis_option == 'Block Limit Warps vs Performance':
    st.header('Block Limit Warps vs Performance')
    scatter_plot('blockLimitWarps', 'durationUsecond', size='blockLimitWarps', title='Block Limit Warps vs Execution Time')

elif analysis_option == 'Block Limit Shared Memory vs Registers':
    st.header('Block Limit Shared Memory vs Registers')
    scatter_plot('blockLimitSharedMem', 'registersPerThread', size='blockLimitSharedMem', title='Block Limit Shared Memory vs Registers Per Thread')

elif analysis_option == 'Achieved Warps vs Occupancy':
    st.header('Achieved Warps vs Occupancy')
    scatter_plot('achievedActiveWarpsPerSm', 'achievedOccupancyPercent', size='achievedActiveWarpsPerSm', title='Achieved Warps vs Occupancy')

elif analysis_option == 'SM Utilization vs Memory Throughput':
    st.header('SM Utilization vs Memory Throughput')
    scatter_plot('smActiveCycles', 'memoryThroughputPercent', size='smActiveCycles', title='SM Utilization vs Memory Throughput')

elif analysis_option == 'Compute Throughput vs DRAM Throughput':
    st.header('Compute Throughput vs DRAM Throughput')
    scatter_plot('computeSMThroughputPercent', 'dramThroughputPercent', size='durationUsecond', title='Compute Throughput vs DRAM Throughput')

elif analysis_option == 'Cache Throughput vs SM Active Cycles':
    st.header('Cache Throughput vs SM Active Cycles')
    scatter_plot('l1TexCacheThroughputPercent', 'smActiveCycles', size='durationUsecond', title='Cache Throughput vs SM Active Cycles')

# Adding an option to easily add new graphs for custom block size performance analysis
st.sidebar.subheader('Add New Graph')
new_x_axis = st.sidebar.selectbox('Choose X-axis', data.columns)
new_y_axis = st.sidebar.selectbox('Choose Y-axis', data.columns)
add_new_graph = st.sidebar.button('Generate Custom Graph')

if add_new_graph:
    st.header(f'{new_x_axis} vs {new_y_axis}')
    scatter_plot(new_x_axis, new_y_axis, title=f'{new_x_axis} vs {new_y_axis}')
