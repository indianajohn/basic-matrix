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
  std::vector<size_t> result_indices;
  result_indices.push_back(rand() % points.size());
  std::unordered_set<size_t> selected_set;
  while (result_indices.size() < num_points &&
         result_indices.size() < points.size()) {
    double max_min_distance = 0;
    size_t max_idx = 0;
    // Try num_attempts times for each centroid, save the one that yields
    // the largest min distance from all the centroids.
    size_t attempt_num = 0;
    while (attempt_num < num_attempts) {
      size_t candidate_idx = rand() % points.size();
      while (selected_set.find(candidate_idx) != selected_set.end()) {
        candidate_idx = rand() % points.size();
      }
      auto point = points[candidate_idx];
      double min_distance = std::numeric_limits<double>::max();
      for (const auto &existing_centroid_idx : result_indices) {
        auto existing_centroid_pt = points[existing_centroid_idx];
        double distance = (existing_centroid_pt - point).norm();
        if (distance < min_distance) {
          min_distance = distance;
        }
      }
      if (min_distance > max_min_distance) {
        max_idx = candidate_idx;
        max_min_distance = min_distance;
      }
      attempt_num++;
    }
    selected_set.insert(max_idx);
    result_indices.push_back(max_idx);
  }
  for (const auto &idx : result_indices) {
    result.push_back(KMeansCluster(points[idx]));
  }
  return result;
}

void assignNearestClusterCenters(const std::vector<Matrix> &points,
                                 KMeansClustering &result) {
  // Clear all cluster members
  for (auto &cluster : result.clusters) {
    cluster.member_indices.clear();
  }
  // Assign points to nearest centroid
  for (size_t i = 0; i < points.size(); i++) {
    const Matrix &point = points[i];
    size_t nearest_idx = 0;
    double lowest_distance = std::numeric_limits<double>::max();
    for (size_t cluster_idx = 0; cluster_idx < result.clusters.size();
         cluster_idx++) {
      const auto &cluster = result.clusters[cluster_idx];
      double distance = (cluster.centroid - point).norm();
      if (distance < lowest_distance) {
        lowest_distance = distance;
        nearest_idx = cluster_idx;
      }
    }
    result.clusters[nearest_idx].member_indices.push_back(i);
  }
}

double aggregateMovement(const KMeansClustering &c1,
                         const KMeansClustering &c2) {
  double sum = 0.;
  for (size_t i = 0; i < c1.clusters.size(); i++) {
    sum += (c1.clusters[i].centroid - c2.clusters[i].centroid).norm();
  }
  return sum;
}
void recalculateCentroid(KMeansClustering &clustering,
                         const std::vector<Matrix> &points) {
  for (auto &cluster : clustering.clusters) {
    if (cluster.member_indices.size() == 0) {
      continue;
    }
    Matrix centroid = points[cluster.member_indices[0]];
    for (size_t i = 1; i < cluster.member_indices.size(); i++) {
      centroid = centroid + points[cluster.member_indices[i]];
    }
    cluster.centroid =
        centroid * (1.0 / static_cast<double>(cluster.member_indices.size()));
  }
}

}; // namespace
KMeansClustering clusterByNaiveKMeans(const std::vector<Matrix> &points,
                                      const KMeansOptions &options) {
  KMeansClustering result;
  result.clusters = pickInitialClusters(points, options.k);
  result.movement = std::numeric_limits<size_t>::max();
  while (result.movement > options.movement_threshold &&
         result.num_iterations < options.max_iterations) {
    assignNearestClusterCenters(points, result);
    recalculateCentroid(result, points);
    result.num_iterations++;
  }
  return result;
}
}; // namespace k_means
}; // namespace basic_matrix
