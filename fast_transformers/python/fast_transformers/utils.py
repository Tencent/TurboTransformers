import fast_transformers.fast_transformers_cxx as cxx
import contextlib
from fast_transformers.fast_transformers_cxx import set_num_threads

__all__ = ['gperf_guard', 'set_num_threads']


@contextlib.contextmanager
def gperf_guard(filename: str):
    cxx.enable_gperf(filename)
    yield
    cxx.disable_gperf()
