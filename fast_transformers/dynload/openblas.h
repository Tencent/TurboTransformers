#pragma once
#include <memory>
namespace fast_transformers {
namespace dynload {

class Openblas {
public:
  explicit Openblas(const char *path);
  ~Openblas();

  Openblas(Openblas &&o);

  Openblas &operator=(Openblas &&o);

private:
  struct Impl;
  std::unique_ptr<Impl> m_;
};

} // namespace dynload
} // namespace fast_transformers
