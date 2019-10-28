#include "core/tensor.h"
#include "gtest/gtest.h"

namespace fast_transformers {
namespace core {

class tensorTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(tensorTest, inialization) {
    Tensor test_tensor(details::CreateDLPackTensor<float>({3,4}));
    float* buff = test_tensor.mutableData<float>();
    for(int i = 0; i < 12; ++i)
        buff[i] = i * 0.1;
    for(int i = 0; i < 12; ++i)
        ASSERT_FLOAT_EQ(test_tensor.data<float>()[i], i * 0.1);       
}


}//core
}//fast_transformers
