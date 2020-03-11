# Copyright 2020 Tencent
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    # `easy_transformers_cxxd` is the name on debug mode
    import easy_transformers.easy_transformers_cxxd as cxx
except ImportError:
    import easy_transformers.easy_transformers_cxx as cxx
import contextlib

__all__ = ['gperf_guard', 'set_num_threads']

set_num_threads = cxx.set_num_threads


@contextlib.contextmanager
def gperf_guard(filename: str):
    cxx.enable_gperf(filename)
    yield
    cxx.disable_gperf()
