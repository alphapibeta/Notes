#ifndef MATMUL_H
#define MATMUL_H

#include <vector>

void vectorAdd(const int* a, const int* b, int* c, int N);
void verify_result(std::vector<int>& a, std::vector<int>& b, std::vector<int>& c);
// void vector_init(int* vectors, int N){

//     for (int i = 0; i < N; i++) {
//         a[i] = rand() % 100;
//         b[i] = rand() % 100;
//     }
// }
#endif // MATMUL_H