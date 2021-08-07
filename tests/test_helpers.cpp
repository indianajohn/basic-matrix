#include "test_helpers.hpp"
#include <filesystem>
#include <random>
#include <string>

namespace {
auto generateRandomAlphaString(std::size_t len) -> std::string {
  const std::string chars = "0123456789"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "abcdefghijklmnopqrstuvwxyz";
  thread_local auto rng = std::default_random_engine();
  auto dist = std::uniform_int_distribution{{}, chars.size() - 1};
  auto result = std::string(len, '\0');
  std::generate_n(begin(result), len, [&]() { return chars[dist(rng)]; });
  return result;
}
}; // namespace

namespace basic_matrix {
TempDirectory::TempDirectory()
    : m_path(std::filesystem::temp_directory_path() /
             generateRandomAlphaString(16)) {
  std::filesystem::create_directories(m_path);
}

TempDirectory::~TempDirectory() { std::filesystem::remove_all(m_path); }

std::filesystem::path TempDirectory::path() const { return m_path; }
}; // namespace basic_matrix
