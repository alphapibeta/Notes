#ifndef STORAGE_H
#define STORAGE_H

#include <memory>

template<typename T>
class Storage {
public:
    Storage(int size);
    ~Storage();
    T& operator[](int index);
    const T& operator[](int index) const;
    void fill(T value);

private:
    std::unique_ptr<T[]> data;
    int size;
};

#endif // STORAGE_H
