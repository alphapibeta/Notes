// #include <iostream>
// #include <chrono>
// #include <vector>
// #include <cstdlib>
// #include <cmath>  // For fabs (absolute value)
// #include "HessianInversionCPU.h"
// #include "HessianInversionGPU.h"

// using namespace std;
// using namespace std::chrono;

// // Function to generate a random matrix
// vector<vector<float>> generateRandomMatrix(int size) {
//     vector<vector<float>> matrix(size, vector<float>(size));
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             matrix[i][j] = static_cast<float>(rand()) / RAND_MAX * 100;
//         }
//     }
//     return matrix;
// }

// // Function to convert a 2D matrix to a 1D array for GPU processing
// vector<float> flattenMatrix(const vector<vector<float>>& matrix) {
//     int size = matrix.size();
//     vector<float> flat_matrix(size * size);
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             flat_matrix[i * size + j] = matrix[i][j];
//         }
//     }
//     return flat_matrix;
// }

// // Function to print a 2D matrix (for debugging)
// void printMatrix(const vector<vector<float>>& matrix) {
//     for (const auto& row : matrix) {
//         for (float val : row) {
//             cout << val << " ";
//         }
//         cout << endl;
//     }
// }

// // Function to print a 1D matrix as 2D (for debugging GPU output)
// void printFlatMatrix(const vector<float>& flat_matrix, int size) {
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             cout << flat_matrix[i * size + j] << " ";
//         }
//         cout << endl;
//     }
// }

// // Function to compare CPU and GPU results with a tolerance
// bool compareMatrices(const vector<vector<float>>& cpu_matrix, const vector<float>& gpu_matrix, int size, float tolerance = 1e-4) {
//     int mismatches = 0;
//     float max_diff = 0.0f;
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             float cpu_val = cpu_matrix[i][j];
//             float gpu_val = gpu_matrix[i * size + j];
//             float diff = fabs(cpu_val - gpu_val);
//             if (diff > tolerance) {
//                 mismatches++;
//                 max_diff = max(max_diff, diff);
//                 if (mismatches <= 5) {  // Print only first 5 mismatches
//                     cout << "Mismatch at (" << i << ", " << j << "): CPU = " << cpu_val << ", GPU = " << gpu_val << ", diff = " << diff << endl;
//                 }
//             }
//         }
//     }
//     cout << "Total mismatches: " << mismatches << ", Max difference: " << max_diff << endl;
//     return mismatches == 0;
// }



// bool compareMatricesRelative(const vector<vector<float>>& cpu_matrix, const vector<float>& gpu_matrix, int size, float tolerance = 1e-5) {
//     int mismatches = 0;
//     float max_relative_diff = 0.0f;
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             float cpu_val = cpu_matrix[i][j];
//             float gpu_val = gpu_matrix[i * size + j];
//             float abs_diff = fabs(cpu_val - gpu_val);
//             float relative_diff = abs_diff / max(fabs(cpu_val), fabs(gpu_val));
//             if (relative_diff > tolerance) {
//                 mismatches++;
//                 max_relative_diff = max(max_relative_diff, relative_diff);
//                 if (mismatches <= 5) {  // Print only first 5 mismatches
//                     cout << "Mismatch at (" << i << ", " << j << "): CPU = " << cpu_val << ", GPU = " << gpu_val << ", relative diff = " << relative_diff << endl;
//                 }
//             }
//         }
//     }
//     cout << "Total mismatches: " << mismatches << ", Max relative difference: " << max_relative_diff << endl;
//     return mismatches == 0;
// }



// bool compareMatricesHybrid(const vector<vector<float>>& cpu_matrix, const vector<float>& gpu_matrix, int size, float abs_tolerance = 1e-5, float rel_tolerance = 0.01) {
//     int mismatches = 0;
//     float max_abs_diff = 0.0f;
//     float max_rel_diff = 0.0f;
//     float sum_abs_diff = 0.0f;
//     float sum_rel_diff = 0.0f;

//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             float cpu_val = cpu_matrix[i][j];
//             float gpu_val = gpu_matrix[i * size + j];
//             float abs_diff = fabs(cpu_val - gpu_val);
//             float rel_diff = abs_diff / max(fabs(cpu_val), fabs(gpu_val));
            
//             sum_abs_diff += abs_diff;
//             sum_rel_diff += rel_diff;

//             if (abs_diff > abs_tolerance && rel_diff > rel_tolerance) {
//                 mismatches++;
//                 max_abs_diff = max(max_abs_diff, abs_diff);
//                 max_rel_diff = max(max_rel_diff, rel_diff);
//                 if (mismatches <= 5) {
//                     cout << "Mismatch at (" << i << ", " << j << "): CPU = " << cpu_val 
//                          << ", GPU = " << gpu_val << ", abs diff = " << abs_diff 
//                          << ", rel diff = " << rel_diff << endl;
//                 }
//             }
//         }
//     }

//     float avg_abs_diff = sum_abs_diff / (size * size);
//     float avg_rel_diff = sum_rel_diff / (size * size);

//     cout << "Total mismatches: " << mismatches 
//          << "\nMax absolute difference: " << max_abs_diff 
//          << "\nMax relative difference: " << max_rel_diff
//          << "\nAverage absolute difference: " << avg_abs_diff
//          << "\nAverage relative difference: " << avg_rel_diff << endl;

//     return mismatches == 0;
// }





// int main(int argc, char* argv[]) {
//     // Check for matrix size argument
//     if (argc < 2) {
//         cout << "Usage: " << argv[0] << " <matrix_size>" << endl;
//         return 1;
//     }

//     // Get matrix size from command line
//     int matrix_size = std::stoi(argv[1]);

//     // Generate random Hessian matrix
//     vector<vector<float>> matrix = generateRandomMatrix(matrix_size);

//     // Add regularization to stabilize the matrix (for better conditioning)
//     for (int i = 0; i < matrix_size; ++i) {
//         matrix[i][i] += 1e-6;  // Add small regularization to the diagonal
//     }

//     // Print original matrix (for debugging)
//     cout << "Original Matrix:" << endl;
//     // printMatrix(matrix);

//     // CPU Inversion
//     HessianInversionCPU cpu_solver(matrix_size);
//     cpu_solver.setMatrix(matrix);
    
//     auto start_cpu = high_resolution_clock::now();
//     cpu_solver.invert();
//     auto end_cpu = high_resolution_clock::now();
    
//     vector<vector<float>> cpu_inverse = cpu_solver.getInverse();
//     auto duration_cpu = duration_cast<milliseconds>(end_cpu - start_cpu);
//     cout << "CPU Inversion Time: " << duration_cpu.count() << " ms" << endl;

//     // Print CPU Inverse Matrix
//     cout << "CPU Inverse Matrix:" << endl;
//     // printMatrix(cpu_inverse);

//     // Convert matrix to flat array for GPU processing
//     vector<float> flat_matrix = flattenMatrix(matrix);

//     // GPU Inversion
//     HessianInversionGPU gpu_solver(matrix_size);
//     gpu_solver.setMatrix(flat_matrix);

//     auto start_gpu = high_resolution_clock::now();
//     gpu_solver.invert();
//     auto end_gpu = high_resolution_clock::now();
    
//     vector<float> gpu_inverse = gpu_solver.getInverse();
//     auto duration_gpu = duration_cast<milliseconds>(end_gpu - start_gpu);
//     cout << "GPU Inversion Time: " << duration_gpu.count() << " ms" << endl;

//     // Print GPU Inverse Matrix
//     cout << "GPU Inverse Matrix:" << endl;
//     // printFlatMatrix(gpu_inverse, matrix_size);

//     // Compare CPU and GPU results
//     bool are_equal = compareMatrices(cpu_inverse, gpu_inverse, matrix_size);
//     if (are_equal) {
//         cout << "CPU and GPU results match!" << endl;
//     } else {
//         cout << "CPU and GPU results do NOT match!" << endl;
//     }


//     // bool are_equalr = compareMatricesRelative(cpu_inverse, gpu_inverse, matrix_size);
//     // if (are_equalr) {
//     //     cout << "compareMatricesRelative-CPU and GPU results match!" << endl;
//     // } else {
//     //     cout << "compareMatricesRelative-CPU and GPU results do NOT match!" << endl;
//     // }

//     bool are_equalh = compareMatricesHybrid(cpu_inverse, gpu_inverse, matrix_size);
//     if (are_equalh) {
//         cout << "compareMatricesRelative-CPU and GPU results match!" << endl;
//     } else {
//         cout << "compareMatricesRelative-CPU and GPU results do NOT match!" << endl;
//     }
    

//     return 0;
// }


#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cmath>  
#include <typeinfo>
#include "HessianInversionCPU.h"
#include "HessianInversionGPU.h"

using namespace std;
using namespace std::chrono;

template <typename T>
vector<vector<T>> generateRandomMatrix(int size) {
    vector<vector<T>> matrix(size, vector<T>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = static_cast<T>(rand()) / RAND_MAX * 100;
        }
    }
    return matrix;
}

template <typename T>
vector<T> flattenMatrix(const vector<vector<T>>& matrix) {
    int size = matrix.size();
    vector<T> flat_matrix(size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            flat_matrix[i * size + j] = matrix[i][j];
        }
    }
    return flat_matrix;
}

template <typename T>
void printMatrix(const vector<vector<T>>& matrix) {
    for (const auto& row : matrix) {
        for (T val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

template <typename T>
void printFlatMatrix(const vector<T>& flat_matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cout << flat_matrix[i * size + j] << " ";
        }
        cout << endl;
    }
}

template <typename T>
bool compareMatricesHybrid(const vector<vector<T>>& cpu_matrix, const vector<T>& gpu_matrix, int size, T abs_tolerance = 1e-5, T rel_tolerance = 0.01) {
    int mismatches = 0;
    T max_abs_diff = 0;
    T max_rel_diff = 0;
    T sum_abs_diff = 0;
    T sum_rel_diff = 0;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            T cpu_val = cpu_matrix[i][j];
            T gpu_val = gpu_matrix[i * size + j];
            T abs_diff = std::abs(cpu_val - gpu_val);
            T rel_diff = abs_diff / std::max(std::abs(cpu_val), std::abs(gpu_val));
            
            sum_abs_diff += abs_diff;
            sum_rel_diff += rel_diff;

            if (abs_diff > abs_tolerance && rel_diff > rel_tolerance) {
                mismatches++;
                max_abs_diff = std::max(max_abs_diff, abs_diff);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
                if (mismatches <= 5) {
                    cout << "Mismatch at (" << i << ", " << j << "): CPU = " << cpu_val 
                         << ", GPU = " << gpu_val << ", abs diff = " << abs_diff 
                         << ", rel diff = " << rel_diff << endl;
                }
            }
        }
    }

    T avg_abs_diff = sum_abs_diff / (size * size);
    T avg_rel_diff = sum_rel_diff / (size * size);

    cout << "Total mismatches: " << mismatches 
         << "\nMax absolute difference: " << max_abs_diff 
         << "\nMax relative difference: " << max_rel_diff
         << "\nAverage absolute difference: " << avg_abs_diff
         << "\nAverage relative difference: " << avg_rel_diff << endl;

    return mismatches == 0;
}

template <typename T>
void runGPUInversion(int matrix_size, bool run_cpu) {
    cout << "Running GPU Hessian Inversion with " << typeid(T).name() << " precision" << endl;
    cout << "Matrix size: " << matrix_size << "x" << matrix_size << endl;

    vector<vector<T>> matrix = generateRandomMatrix<T>(matrix_size);
    for (int i = 0; i < matrix_size; ++i) {
        matrix[i][i] += 1e-6;  // Add a small value to the diagonal for stability
    }

    vector<T> flat_matrix = flattenMatrix(matrix);
    HessianInversionGPU<T> gpu_solver(matrix_size);
    gpu_solver.setMatrix(flat_matrix);

    auto start_gpu = high_resolution_clock::now();
    gpu_solver.invert();
    auto end_gpu = high_resolution_clock::now();

    vector<T> gpu_inverse = gpu_solver.getInverse();
    auto duration_gpu = duration_cast<milliseconds>(end_gpu - start_gpu);

    cout << "GPU Inversion Time: " << duration_gpu.count() << " ms" << endl;
    cout << "Effective Memory Throughput: " << gpu_solver.getGPUMemoryThroughput() << " GB/s" << endl;
    cout << "GPU Computational Throughput: " << gpu_solver.getGPUComputationalThroughput() << " GFLOP/s" << endl;
    cout << "Arithmetic Intensity: " << gpu_solver.getArithmeticIntensity() << " FLOP/byte" << endl;
    cout << "Theoretical Memory Bandwidth: " << gpu_solver.getTheoreticalGPUBandwidth() << " GB/s" << endl;
    cout << "Theoretical Computational Throughput: " << gpu_solver.getTheoreticalGPUComputationalThroughput() << " GFLOP/s" << endl;

    if (run_cpu) {
        HessianInversionCPU<T> cpu_solver(matrix_size);
        cpu_solver.setMatrix(matrix);

        auto start_cpu = high_resolution_clock::now();
        cpu_solver.invert();
        auto end_cpu = high_resolution_clock::now();

        vector<vector<T>> cpu_inverse = cpu_solver.getInverse();
        auto duration_cpu = duration_cast<milliseconds>(end_cpu - start_cpu);

        cout << "CPU Inversion Time: " << duration_cpu.count() << " ms" << endl;

        bool are_equal = compareMatricesHybrid(cpu_inverse, gpu_inverse, matrix_size);
        if (are_equal) {
            cout << "CPU and GPU results match within tolerance!" << endl;
        } else {
            cout << "CPU and GPU results do NOT match within tolerance!" << endl;
        }
    }

    cout << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <matrix_size> [--cpu]" << endl;
        return 1;
    }

    int matrix_size = std::stoi(argv[1]);
    bool run_cpu = false;

    // Check if --cpu flag is present
    for (int i = 2; i < argc; ++i) {
        if (string(argv[i]) == "--cpu") {
            run_cpu = true;
            break;
        }
    }

    // Run with float precision
    // runGPUInversion<float>(matrix_size, run_cpu);

    // Run with double precision
    runGPUInversion<double>(matrix_size, run_cpu);

    return 0;
}