#include "k_means.hpp"
#include "matrix_helpers.hpp"

using namespace basic_matrix;

void generatePointSet(std::vector<Matrix> &points,
                      std::vector<Matrix> &true_centroids,
                      const size_t &n_centroids = 5,
                      const size_t &n_points = 500,
                      const double &centroid_diameter = 10.0,
                      const double &field_diameter = 200.0) {
  points.clear();
  true_centroids.clear();

  // Get n_centroids centroids mutually min_distance apart.
  bool sufficient_distance = true;
  double min_distance = sqrt(2) * centroid_diameter;
  while (true_centroids.size() < n_centroids) {
    auto candidate = randomMatrix(2, 1, 0., field_diameter);
    for (const auto &centroid : true_centroids) {
      if ((centroid - candidate).norm() < min_distance) {
        sufficient_distance = false;
      }
    }
    if (sufficient_distance) {
      true_centroids.push_back(candidate);
    }
  }
  // For each sample point, select a centroid at random, select a vector
  // at random, add the two, and add the point. Points should appear in
  // a uniform distribution around the centroids. A small percentage of
  // points will be outliers.
  while (points.size() < n_points) {
    auto centroid = true_centroids[rand() % n_centroids];
    // outlier - belongs to no cluster.
    if (rand() % 100 < 2) {
      centroid = randomMatrix(2, 1, 0., 100.);
    }
    points.push_back(centroid +
                     randomMatrix(2, 1, -centroid_diameter, centroid_diameter));
  }
}

void naiveKMeansE2E() {
  std::vector<Matrix> points;
  std::vector<Matrix> true_centroids;
  generatePointSet(points, true_centroids);
  k_means::KMeansOptions options;
  auto result = k_means::clusterByNaiveKMeans(points, options);
  ASSERT_EQ(result.clusters.size(), options.k);
  size_t num_points = 0;
  for (const auto &cluster : result.clusters) {
    num_points += cluster.member_indices.size();
  }
  ASSERT_EQ(num_points, points.size());
  for (const auto &cluster : result.clusters) {
    double min_distance = std::numeric_limits<double>::max();
    for (const auto &centroid : true_centroids) {
      double distance = (centroid - points[cluster.centroid]).norm();
      if (distance < min_distance) {
        min_distance = distance;
      }
    }
    std::cout << "min_distance=" << min_distance << std::endl;
  }
}

int main() {
  naiveKMeansE2E();
  return 0;
}
