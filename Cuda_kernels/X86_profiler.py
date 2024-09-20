import subprocess
import pandas as pd
import re
import argparse
import os
import signal

# Function to extract duration (support for microseconds and milliseconds)
def extract_duration(output):
    """Extracts duration in microseconds, converts from milliseconds if necessary."""
    usec_match = re.search(r"Duration\s+usecond\s+([\d.]+)", output)
    msec_match = re.search(r"Duration\s+msecond\s+([\d.]+)", output)
    sec_match = re.search(r"Duration\s+second\s+([\d.]+)", output)
    if usec_match:
        return float(usec_match.group(1))  # microseconds
    elif msec_match:
        return float(msec_match.group(1)) * 1000  # Convert milliseconds to microseconds
    elif sec_match:
        return float(sec_match.group(1)) * 1000000  # Convert milliseconds to microseconds
    return "N/A"

def parse_kernel_args(kernel_args):
    parsed_args = {}
    for arg in kernel_args:
        key, value = arg.split("=")
        parsed_args[key] = value
    return parsed_args

# Function to extract metrics with consideration for unit conversion and format float values to 4 decimal places
def extract_metric_with_unit(output, regex, data_type="float", unit="default"):
    """Extracts metrics with unit conversion."""
    match = re.search(regex, output)
    if match:
        value = match.group(1).replace(",", "")
        
        if data_type == "int":
            return int(value)
        elif data_type == "unsigned int":
            return abs(int(value))
        elif data_type == "float":
            if unit == "Kbyte" and "byte" in regex:
                return "%.6f" % (float(value) / 1024)  # Convert byte to Kbyte
            elif unit == "byte" and "Kbyte" in regex:
                return "%.6f" % (float(value) * 1024)  # Convert Kbyte to byte
            elif "Kbyte/block" in regex and unit == "byte/block":
                return "%.6f" % (float(value) * 1024)  # Convert Kbyte/block to byte/block
            elif "byte/block" in regex and unit == "Kbyte/block":
                return "%.6f" % (float(value) / 1024)  # Convert byte/block to Kbyte/block
            return "%.6f" % float(value)
        elif data_type == "string":
            return value  # Return string values as is
    return "N/A"

# Function to extract WRN/INF/OPT multi-line messages
def extract_wrn_inf_opt(output):
    """Extracts multi-line WRN, INF, and OPT messages."""
    wrn_matches = re.findall(r"(WRN.*?)(?=\n\n|\Z)", output, re.DOTALL)
    inf_matches = re.findall(r"(INF.*?)(?=\n\n|\Z)", output, re.DOTALL)
    opt_matches = re.findall(r"(OPT.*?)(?=\n\n|\Z)", output, re.DOTALL)
    return "\n".join(wrn_matches) if wrn_matches else "N/A", \
           "\n".join(inf_matches) if inf_matches else "N/A", \
           "\n".join(opt_matches) if opt_matches else "N/A"

# Function to parse the Nsight Compute output and extract relevant metrics
def parse_ncu_output(output, kernel_name, build_dir, block_thread_x, block_thread_y):
    """Parses the Nsight Compute output and returns a dictionary with parsed metrics."""
    wrn_messages, inf_messages, opt_messages = extract_wrn_inf_opt(output)

    data = {
        "buildDirectory": build_dir,
        "kernelName": kernel_name,
        "blockSizeX": block_thread_x,
        "blockSizeY": block_thread_y,
        "totalThreads": block_thread_x * block_thread_y if block_thread_x and block_thread_y else "N/A",
        
        "smFrequencyCyclePerUsecond": extract_metric_with_unit(output, r"SM Frequency\s+cycle/(?:usecond|nsecond|second)\s+([\d.]+)", "float"),
        "elapsedCycles": extract_metric_with_unit(output, r"Elapsed Cycles\s+cycle\s+([\d,]+)", "unsigned int"),
        "memoryThroughputPercent": extract_metric_with_unit(output, r"Memory Throughput\s+%\s+([\d.]+)", "float"),
        "dramThroughputPercent": extract_metric_with_unit(output, r"DRAM Throughput\s+%\s+([\d.]+)", "float"),
        "durationUsecond": extract_duration(output),
        "l1TexCacheThroughputPercent": extract_metric_with_unit(output, r"L1/TEX Cache Throughput\s+%\s+([\d.]+)", "float"),
        "l2CacheThroughputPercent": extract_metric_with_unit(output, r"L2 Cache Throughput\s+%\s+([\d.]+)", "float"),
        "smActiveCycles": extract_metric_with_unit(output, r"SM Active Cycles\s+cycle\s+([\d,]+)", "unsigned int"),
        "computeSMThroughputPercent": extract_metric_with_unit(output, r"Compute \(SM\) Throughput\s+%\s+([\d.]+)", "float"),

        # Launch Statistics Section
        "blockSize": extract_metric_with_unit(output, r"Block Size\s+(\d+)", "int"),
        "gridSize": extract_metric_with_unit(output, r"Grid Size\s+([\d,]+)", "unsigned int"),
        "registersPerThread": extract_metric_with_unit(output, r"Registers Per Thread\s+register/thread\s+([\d]+)", "unsigned int"),
        "sharedMemoryConfigSizeKbyte": extract_metric_with_unit(output, r"Shared Memory Configuration Size\s+(?:byte|Kbyte)\s+([\d.]+)", "float", "Kbyte"),
        "driverSharedMemoryPerBlockByte": extract_metric_with_unit(output, r"Driver Shared Memory Per Block\s+byte/block\s+([\d,]+)", "unsigned int"),
        "dynamicSharedMemoryPerBlockKbyte": extract_metric_with_unit(output, r"Dynamic Shared Memory Per Block\s+(?:byte|Kbyte)/block\s+([\d.]+)", "float", "Kbyte/block"),
        "staticSharedMemoryPerBlockByte": extract_metric_with_unit(output, r"Static Shared Memory Per Block\s+byte/block\s+([\d,]+)", "unsigned int"),
        "wavesPerSM": extract_metric_with_unit(output, r"Waves Per SM\s+([\d.]+)", "float"),  # Fix: Now treated as float

        # Occupancy Section
        "blockLimitSm": extract_metric_with_unit(output, r"Block Limit SM\s+block\s+([\d]+)", "unsigned int"),
        "blockLimitRegisters": extract_metric_with_unit(output, r"Block Limit Registers\s+block\s+([\d]+)", "unsigned int"),
        "blockLimitSharedMem": extract_metric_with_unit(output, r"Block Limit Shared Mem\s+block\s+([\d]+)", "unsigned int"),
        "blockLimitWarps": extract_metric_with_unit(output, r"Block Limit Warps\s+block\s+([\d]+)", "unsigned int"),
        "theoreticalActiveWarpsPerSm": extract_metric_with_unit(output, r"Theoretical Active Warps per SM\s+warp\s+([\d]+)", "unsigned int"),
        "theoreticalOccupancyPercent": extract_metric_with_unit(output, r"Theoretical Occupancy\s+%\s+([\d.]+)", "float"),
        "achievedOccupancyPercent": extract_metric_with_unit(output, r"Achieved Occupancy\s+%\s+([\d.]+)", "float"),
        "achievedActiveWarpsPerSm": extract_metric_with_unit(output, r"Achieved Active Warps Per SM\s+warp\s+([\d.]+)", "float"),

        # WRN/INF/OPT messages
        "warningsWrn": wrn_messages,
        "informationInf": inf_messages,
        "optimizationHintsOpt": opt_messages,
    }

    return data

# Function to run Nsight Compute for a given kernel and build directory
def run_ncu_and_parse(build_dir, kernel_name, exec_name, kernel_args, block_thread_x=None, block_thread_y=None, output_file=None):
    # Ensure the path to the executable is correct
    exec_path = os.path.join(build_dir, exec_name)

    if not os.path.isfile(exec_path) or not os.access(exec_path, os.X_OK):
        print(f"ERROR: {exec_path} does not exist or is not an executable.")
        return {}

    print(f"Running Nsight Compute for kernel: {kernel_name} in directory: {build_dir} with args: {kernel_args}")

    # Create dynamic kernel argument string
    dynamic_kernel_args = []
    for key, value in kernel_args.items():
        if key == "block-x" and block_thread_x is not None:
            value = str(block_thread_x)  # Replace block-x with loop value
        elif key == "block-y" and block_thread_y is not None:
            value = str(block_thread_y)  # Replace block-y with loop value
        dynamic_kernel_args.append(value)

    # Create the final kernel argument string
    kernel_args_dynamic = " ".join(dynamic_kernel_args)
    
    # Build the ncu command for X86
    ncu_cmd = f"ncu --kernel-name {kernel_name} --launch-skip 0 --launch-count 1 {exec_path} {kernel_args_dynamic}"
    
    try:
        # Set a timeout for the ncu command
        ncu_output = subprocess.check_output(ncu_cmd, shell=True, timeout=120, text=True)  # Timeout after 120 seconds
    except subprocess.TimeoutExpired:
        print(f"ERROR: Nsight Compute for {kernel_name} timed out.")
        return {}
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Nsight Compute for {kernel_name} failed with error: {e}")
        return {}

    # Write the Nsight Compute output to a text file, separating each execution
    if output_file:
        try:
            with open(output_file, "a") as f:
                f.write(f"Running Nsight Compute for kernel: {kernel_name} in directory: {build_dir} with args: {kernel_args_dynamic}\n")
                f.write(ncu_output)
                f.write("\n" + "=" * 100 + "\n")
        except IOError as e:
            print(f"ERROR: Failed to write Nsight Compute output to {output_file}: {e}")
    
    # Extract relevant sections from the output and return the parsed data
    return parse_ncu_output(ncu_output, kernel_name, build_dir, block_thread_x, block_thread_y)

# Main function to profile multiple kernels with dynamic block sizes and kernel args
def profile_kernels_with_sizes(build_dirs, kernel_names, exec_name, output_file, kernel_args):
    results = []

    # Parse kernel arguments into a dictionary
    parsed_kernel_args = parse_kernel_args(kernel_args)

    # Loop through each build directory and kernel name
    for build_dir in build_dirs:
        for kernel_name in kernel_names:
            # Loop through different block sizes, ensuring total threads do not exceed 1024
            for block_thread_x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]: # [8,16,64]: # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                for block_thread_y in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:# [256,128,64]: #[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                    total_threads = block_thread_x * block_thread_y
                    if total_threads <= 1024:
                        # Run Nsight Compute with the parsed and dynamic kernel arguments
                        result = run_ncu_and_parse(build_dir, kernel_name, exec_name, parsed_kernel_args, block_thread_x, block_thread_y, output_file)
                        if result:
                            results.append(result)

    # Convert results to a DataFrame for better display and analysis
    df = pd.DataFrame(results)
    
    # Print the results as a table
    print(df.to_string(index=False))
    
    # Save as CSV using the provided output_file argument
    csv_file = f"{output_file}.csv"
    df.to_csv(csv_file, encoding='utf-8', index=False)
   
    # Save as a text file with the same format
    txt_file = f"{output_file}.txt"
    with open(txt_file, "w") as txt_file_handle:
        txt_file_handle.write(df.to_string(index=False))

# Entry point for command-line arguments
if __name__ == "__main__":
    # Argument parser for dynamic input
    parser = argparse.ArgumentParser(description="Nsight Compute Profiler for CUDA Kernels with Dynamic Block Sizes")
    parser.add_argument("--build_dirs", nargs='+', required=True, help="List of build directories to profile")
    parser.add_argument("--kernels", nargs='+', required=True, help="List of kernel names to profile")
    parser.add_argument("--exec_name", default="./heat_solver", help="Name of the executable file")
    parser.add_argument("--kernel_args", nargs='+', help="Kernel arguments to pass (e.g., block-x=32 block-y=32 N=2048 nsteps=1600 v_cpu=0)")
    parser.add_argument("--output_file", default="file", help="File to store Nsight Compute outputs")
    
    args = parser.parse_args()

    # Call the profiling function with dynamic block sizes and store Nsight Compute output to a file
    profile_kernels_with_sizes(args.build_dirs, args.kernels, args.exec_name, args.output_file, args.kernel_args)
