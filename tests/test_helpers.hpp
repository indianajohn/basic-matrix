#include <math.h>
#include <stdexcept>
#include <string>
#define ASSERT(condition)                                                      \
  {                                                                            \
    if (!(condition)) {                                                        \
      throw std::runtime_error(std::string(__FILE__) + std::string(":") +      \
                               std::to_string(__LINE__) +                      \
                               std::string(" in ") +                           \
                               std::string(__PRETTY_FUNCTION__));              \
    }                                                                          \
  }

#define ASSERT_EQ(x, y)                                                        \
  {                                                                            \
    if ((x) != (y)) {                                                          \
      throw std::runtime_error(                                                \
          std::string(__FILE__) + std::string(":") +                           \
          std::to_string(__LINE__) + std::string(" in ") +                     \
          std::string(__PRETTY_FUNCTION__) + std::string(": ") +               \
          std::to_string((x)) + std::string(" != ") + std::to_string((y)));    \
    }                                                                          \
  }

#define ASSERT_TOL(x, y, tol)                                                  \
  {                                                                            \
    if (fabs(x - y) > tol) {                                                   \
      throw std::runtime_error(                                                \
          std::string(__FILE__) + std::string(":") +                           \
          std::to_string(__LINE__) + std::string(" in ") +                     \
          std::string(__PRETTY_FUNCTION__) + std::string(": ") +               \
          std::to_string((x)) + std::string(" not near ") +                    \
          std::to_string((y)));                                                \
    }                                                                          \
  }

#define ASSERT_NEAR(x, y) ASSERT_TOL(x, y, 1e-9)
