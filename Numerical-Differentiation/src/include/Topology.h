#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include <vector>
#include <set>

class Topology {
public:
    Topology(const std::set<int>& setX, const std::vector<std::set<int>>& openSets);
    bool is_open_set(const std::set<int>& set) const;

private:
    std::set<int> X;
    std::vector<std::set<int>> T;
};

#endif // TOPOLOGY_H