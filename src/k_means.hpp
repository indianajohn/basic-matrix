#pragma once
#include "matrix.hpp"

namespace basic_matrix {
namespace k_means {

/// A single cluster.
struct KMeansCluster {
  /// Centroid of the cluster.
  Matrix centroid;
  /// Indices in the original poitn set of points in the cluster.
  std::vector<size_t> member_indices;
};

/// The result of a k-means clustering.
struct KMeansClustering {
  /// Clusters
  std::vector<KMeansCluster> clusters;
};

struct KMeansOptions {
  /// Maximum number of iterations.
  size_t max_iterations = 100;

  /// Number of cluster centers.
  size_t k = 10;

  // Movement center for all thresholds
  double movement_threshold = 0.0;
};

/// Clusters a vector of N-dimensional points (each point much be the
/// same dimensions).
KMeansClustering
clusterByNaiveKMeans(const std::vector<Matrix> &points,
                     const KMeansOptions &options = KMeansOptions());

}; // namespace k_means
}; // namespace basic_matrix
