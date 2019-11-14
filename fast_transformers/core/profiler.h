#pragma once

#include <string>

namespace fast_transformers {
namespace core {

void EnableGPerf(const std::string& profile_file);
void DisableGPerf();

}  // namespace core
}  // namespace fast_transformers
