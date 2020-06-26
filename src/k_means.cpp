#include "k_means.hpp"
#include <unordered_set>

namespace basic_matrix {
namespace k_means {
namespace {
std::vector<KMeansCluster>
pickInitialClusters(const std::vector<Matrix> &points, const size_t &num_points,
                    const size_t &num_attempts = 100) {
  std::vector<KMeansCluster> result;
  if (num_points == 0) {
    return result;
  }
  // Pick one point at random.
  result.push_back(KMeansCluster(rand() % points.size()));
  std::unordered_set<size_t> selected_set;
  size_t attempt_num = 0;
  while (result.size() < num_points && result.size() < points.size()) {
    double max_min_distance = 0;
    size_t max_idx = 0;
    // Try num_attempts times for each centroid, save the one that yields
    // the largest min distance from all the centroids.
    while (attempt_num < num_attempts) {
      size_t candidate_idx = rand() % points.size();
      while (selected_set.find(candidate_idx) != selected_set.end()) {
        candidate_idx = rand() % points.size();
      }
      auto point = points[candidate_idx];
      double min_distance = 0;
      for (const auto &existing_centroid : result) {
        auto existing_centroid_pt = points[existing_centroid.centroid];
        double distance = (existing_centroid_pt - point).norm();
        if (distance < min_distance) {
          distance = min_distance;
        }
      }
      if (min_distance > max_min_distance) {
        max_idx = candidate_idx;
        max_min_distance = min_distance;
      }
      attempt_num++;
    }
    selected_set.insert(max_idx);
    result.push_back(max_idx);
  }
  return result;
}
}; // namespace
KMeansClustering clusterByNaiveKMeans(const std::vector<Matrix> &points,
                                      const KMeansOptions &options) {
  KMeansClustering result;
  result.clusters = pickInitialClusters(points, options.k);
  // TODO
  return result;
}
}; // namespace k_means
}; // namespace basic_matrix
