#include "Topology.h"

Topology::Topology(const std::set<int>& setX, const std::vector<std::set<int>>& openSets) : X(setX), T(openSets) {}

bool Topology::is_open_set(const std::set<int>& set) const {
    for (const auto& openSet : T) {
        if (openSet == set) {
            return true;
        }
    }
    return false;
}