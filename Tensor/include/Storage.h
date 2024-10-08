#ifndef STORAGE_H
#define STORAGE_H
#include <memory>

template<typename T>
class Storage {
public:
    Storage(int size);  // Constructor
    ~Storage();         // Destructor

    T& operator[](int index);  // Access element at index
    const T& operator[](int index) const;  // Const access element at index
    void fill(T value);  // Fill the storage with a value

private:
    std::unique_ptr<T[]> data;  // Smart pointer to manage the array
    int size;
};

#endif  // STORAGE_H
