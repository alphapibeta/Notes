#ifndef DISTANCE_CALCULATOR_H
#define DISTANCE_CALCULATOR_H

class DistanceCalculator {
public:
    DistanceCalculator(int num_points, int threads_x, int threads_y);
    ~DistanceCalculator();
    void setPoints(float* x_coords, float* y_coords);
    void generateRandomPoints();
    void generateSequentialPoints();
    void computeDistances();
    float getMaxDistance();

private:
    int num_points;
    int threads_x, threads_y;
    int num_pairs;
    float* d_x_coords;
    float* d_y_coords;
    float* d_distances;
    float* d_max_distance;
    float* h_max_distance;
    void allocateMemory();
    void freeMemory();
};

#endif