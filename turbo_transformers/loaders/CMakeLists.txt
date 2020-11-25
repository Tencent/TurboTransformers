
add_library(tt_npz_loader npz_load.cpp)
target_link_libraries(tt_npz_loader
        PUBLIC dlpack cnpy zlib
        PRIVATE tt_core
)
