#include "DistanceCalculator.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <nvtx3/nvToolsExt.h>
#include <cstdlib>
#include <ctime>

DistanceCalculator::DistanceCalculator(int num_points, int threads_x, int threads_y)
    : num_points(num_points), threads_x(threads_x), threads_y(threads_y) {
    num_pairs = (num_points * (num_points - 1)) / 2;
    allocateMemory();
}

DistanceCalculator::~DistanceCalculator() {
    freeMemory();
}

void DistanceCalculator::allocateMemory() {
    cudaMalloc((void**)&d_x_coords, num_points * sizeof(float));
    cudaMalloc((void**)&d_y_coords, num_points * sizeof(float));
    cudaMalloc((void**)&d_distances, num_pairs * sizeof(float));
    cudaMalloc((void**)&d_max_distance, sizeof(float));
    h_max_distance = new float;
}

void DistanceCalculator::freeMemory() {
    cudaFree(d_x_coords);
    cudaFree(d_y_coords);
    cudaFree(d_distances);
    cudaFree(d_max_distance);
    delete h_max_distance;
}

void DistanceCalculator::setPoints(float* x_coords, float* y_coords) {
    cudaMemcpy(d_x_coords, x_coords, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_coords, y_coords, num_points * sizeof(float), cudaMemcpyHostToDevice);
}

void DistanceCalculator::generateRandomPoints() {
    float* h_x_coords = new float[num_points];
    float* h_y_coords = new float[num_points];
    srand(time(0));
    for (int i = 0; i < num_points; ++i) {
        h_x_coords[i] = rand() % 100;
        h_y_coords[i] = rand() % 100;
    }
    setPoints(h_x_coords, h_y_coords);
    delete[] h_x_coords;
    delete[] h_y_coords;
}

void DistanceCalculator::generateSequentialPoints() {
    float* h_x_coords = new float[num_points];
    float* h_y_coords = new float[num_points];
    for (int i = 0; i < num_points; ++i) {
        h_x_coords[i] = static_cast<float>(i);
        h_y_coords[i] = static_cast<float>(i);
    }
    setPoints(h_x_coords, h_y_coords);
    delete[] h_x_coords;
    delete[] h_y_coords;
}

float DistanceCalculator::getMaxDistance() {
    cudaMemcpy(h_max_distance, d_max_distance, sizeof(float), cudaMemcpyDeviceToHost);
    return *h_max_distance;
}

extern void launchComputeDistances(float* d_x_coords, float* d_y_coords, float* d_distances, int num_points, int threads_x, int threads_y);
extern void launchReduceMax(float* d_distances, float* d_max_distance, int num_pairs, int threads_x, int threads_y);

void DistanceCalculator::computeDistances() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    launchComputeDistances(d_x_coords, d_y_coords, d_distances, num_points, threads_x, threads_y);
    launchReduceMax(d_distances, d_max_distance, num_pairs, threads_x, threads_y);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}