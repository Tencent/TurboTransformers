#pragma once

#include <string>

namespace fast_transformers {
namespace core {

void EnableGperf(const std::string& profile_file);
void DisableGperf();

}  // namespace core
}  // namespace fast_transformers
