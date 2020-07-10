## How to add an optimized custom layer

Take [multi_headed_attention](https://github.com/OpenNMT/OpenNMT-py/blob/b98fb3d7cb/onmt/modules/multi_headed_attn.py) as an example.

1. implement your layer as in ./layers/multiple_headed_attention.cpp

2. register multiple_headed_attention.cpp in ./layers/CMakeLists.txt

3. add python API in turbo_transformers/python/turbo_transformers/layers/modeling_decoder.py

4. register in ./turbo_transformers/python/turbo_transformers/layers/__init__.py

5. add a `py::class_` in ./turbo_transformers/python/pybind.cpp

6. add an unitest in ./turbo_transformers/python/tests/multi_headed_attention_test.py

