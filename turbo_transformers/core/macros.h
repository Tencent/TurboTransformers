

#pragma once
#include <cfloat>

// Disable the copy and assignment operator for a class.
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)         \
 private:                                          \
  classname(const classname&) = delete;            \
  classname(classname&&) = delete;                 \
  classname& operator=(const classname&) = delete; \
  classname& operator=(classname&&) = delete
#endif

#if defined(__FLT_MAX__)
#define FLT_MAX __FLT_MAX__
#endif  // __FLT_MAX__
