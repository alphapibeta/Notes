#include "Storage.h"

template<typename T>
Storage<T>::Storage(int size) : size(size) {
    // Allocating memory for CPU using smart pointer
    data = std::make_unique<T[]>(size);
}

template<typename T>
Storage<T>::~Storage() {
    // No manual memory management required, unique_ptr will automatically handle it.
}

template<typename T>
T& Storage<T>::operator[](int index) {
    return data[index];
}

template<typename T>
const T& Storage<T>::operator[](int index) const {
    return data[index];
}

template<typename T>
void Storage<T>::fill(T value) {
    for (int i = 0; i < size; ++i) {
        data[i] = value;
    }
}

// Explicit template instantiations
template class Storage<float>;
template class Storage<double>;
