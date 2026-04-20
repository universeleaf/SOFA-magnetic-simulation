#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace elastic_rod_guidewire
{
struct ElasticRodGuidewireParameters
{
    std::vector<std::array<double, 3>> initialNodes;
    std::vector<std::array<double, 3>> undeformedNodes;
    std::vector<std::array<double, 3>> tubeNodes;
    std::vector<double> tubeRadii;

    double rho = 7800.0;
    double rodRadius = 0.18;
    double dt = 0.005;
    double youngHead = 8.0e10;
    double youngBody = 2.1e11;
    double shearHead = 3.0e10;
    double shearBody = 7.9e10;
    double rodLength = 420.0;
    std::size_t magneticEdgeCount = 5;
    std::size_t pushNodeCount = 2;
};
} // namespace elastic_rod_guidewire
