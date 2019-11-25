################################################################################
#
# \file      cmake/FindMKL.cmake
# \author    J. Bakosi
# \copyright 2012-2015, Jozsef Bakosi, 2016, Los Alamos National Security, LLC.
# \brief     Find the Math Kernel Library from Intel
# \date      Thu 26 Jan 2017 02:05:50 PM MST
#
################################################################################

# Find the Math Kernel Library from Intel
#
#  MKL_FOUND - System has MKL
#  MKL_INCLUDE_DIRS - MKL include files directories
#  MKL_LIBRARIES - The MKL libraries
#  MKL_INTERFACE_LIBRARY - MKL interface library
#  MKL_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  MKL_CORE_LIBRARY - MKL core library
#
#  The environment variables MKLROOT and INTEL are used to find the library.
#  Everything else is ignored. If MKL is found "-DMKL_ILP64" is added to
#  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS.
#
#  Example usage:
#
#  find_package(MKL)
#  if(MKL_FOUND)
#    target_link_libraries(TARGET ${MKL_LIBRARIES})
#  endif()

# If already in cache, be silent
if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
    MKL_SEQUENTIAL_LAYER_LIBRARY AND MKL_CORE_LIBRARY)
  set (MKL_FIND_QUIETLY TRUE)
endif()

set(INT_LIB "mkl_intel_lp64")
set(THR_LIB "mkl_intel_thread")
set(COR_LIB "mkl_core")

message(STATUS ${CMAKE_PREFIX_PATH}/include)
find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS $ENV{CONDA_PREFIX}/include ${MKLROOT}/include )

find_library(MKL_INTERFACE_LIBRARY
             NAMES ${INT_LIB}
             PATHS
                   $ENV{CONDA_PREFIX}/lib
                   ${MKLROOT}/lib
                   ${MKLROOT}/lib/intel64
                   ${INTEL}/mkl/lib/intel64)

find_library(MKL_THREADS_LIBRARY
             NAMES ${THR_LIB}
             PATHS
                   $ENV{CONDA_PREFIX}/lib
                   ${MKLROOT}/lib
                   ${MKLROOT}/lib/intel64
                   ${INTEL}/mkl/lib/intel64)

find_library(MKL_CORE_LIBRARY
             NAMES ${COR_LIB}
             PATHS
                   $ENV{CONDA_PREFIX}/lib
                   ${MKLROOT}/lib
                   ${MKLROOT}/lib/intel64
                   ${INTEL}/mkl/lib/intel64)

set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
set(MKL_LIBRARIES ${MKL_INTERFACE_LIBRARY} ${MKL_THREADS_LIBRARY} ${MKL_CORE_LIBRARY})


if (NOT APPLE)
    find_library(IOMP_LIBRARY
            NAMES iomp5
            PATHS
                   $ENV{CONDA_PREFIX}/lib
                   ${MKLROOT}/lib
                   ${MKLROOT}/lib/intel64
                   ${MKLROOT}/../lib
                   ${MKLROOT}/../lib/intel64)

    set(MKL_LIBRARIES ${MKL_LIBRARIES} ${IOMP_LIBRARY})
endif ()


if (MKL_INCLUDE_DIR AND
    MKL_INTERFACE_LIBRARY AND
    MKL_THREADS_LIBRARY AND
    MKL_CORE_LIBRARY)

    if (NOT DEFINED ENV{CRAY_PRGENVPGI} AND
        NOT DEFINED ENV{CRAY_PRGENVGNU} AND
        NOT DEFINED ENV{CRAY_PRGENVCRAY} AND
        NOT DEFINED ENV{CRAY_PRGENVINTEL})
      set(ABI "-m64")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ABI}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ABI}")

else()

  set(MKL_INCLUDE_DIRS "")
  set(MKL_LIBRARIES "")
  set(MKL_INTERFACE_LIBRARY "")
  set(MKL_THREADS_LIBRARY "")
  set(MKL_CORE_LIBRARY "")

endif()

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS MKL_INTERFACE_LIBRARY MKL_THREADS_LIBRARY MKL_CORE_LIBRARY)

MARK_AS_ADVANCED(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_INTERFACE_LIBRARY MKL_THREADS_LIBRARY MKL_CORE_LIBRARY)
