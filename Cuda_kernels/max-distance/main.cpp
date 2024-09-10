#include "DistanceCalculator.h"
#include <iostream>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <threads_x> <threads_y> <num_points> <init_mode>" << std::endl;
        std::cerr << "init_mode: 0 for random, 1 for sequential" << std::endl;
        return 1;
    }

    int threads_x = std::atoi(argv[1]);
    int threads_y = std::atoi(argv[2]);
    int num_points = std::atoi(argv[3]);
    int init_mode = std::atoi(argv[4]);  // 0 for random, 1 for sequential

    // Create an instance of the DistanceCalculator
    DistanceCalculator calculator(num_points, threads_x, threads_y);

    // Initialize points based on the mode
    if (init_mode == 0) {
        calculator.generateRandomPoints();  // Random initialization
    } else if (init_mode == 1) {
        calculator.generateSequentialPoints();  // Sequential initialization
    } else {
        std::cerr << "Invalid initialization mode. Use 0 for random, 1 for sequential." << std::endl;
        return 1;
    }

    nvtxRangePush("Distance Computation");

    // Set points on the GPU and compute the distances
    calculator.computeDistances();

    // Get and print the maximum distance
    float max_distance = calculator.getMaxDistance();
    std::cout << "Maximum distance: " << max_distance << std::endl;

    nvtxRangePop();

    return 0;
}
