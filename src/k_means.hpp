#pragma once
#include "matrix.hpp"

namespace basic_matrix {
namespace k_means {
struct KMeansCluster {
  Matrix centroid;
  std::vector<size_t> member_indices;
};
struct KMeansClustering {
  basic_matrix::Matrix centroid;
  std::vector<KMeansCluster> clusters;
};

struct KMeansOptions {
  size_t max_iterations = 100;
  size_t k = 10;
};

/// Clusters a vector of N-dimensional points (each point much be the
/// same dimensions).
KMeansClustering
clusterByNaiveKMeans(const std::vector<Matrix> &points,
                     const KMeansOptions &options = KMeansOptions());

}; // namespace k_means
}; // namespace basic_matrix
