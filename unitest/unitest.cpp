#include "gtest/gtest.h"

int main(int argc, char **argv) {
    std::setlocale(LC_ALL, "en_US.utf8");

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
