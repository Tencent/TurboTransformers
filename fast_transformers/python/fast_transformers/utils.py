try:
    # `fast_transformers_cxxd` is the name on debug mode
    import fast_transformers.fast_transformers_cxxd as cxx
except ImportError:
    import fast_transformers.fast_transformers_cxx as cxx
import contextlib

__all__ = ['gperf_guard', 'set_num_threads']

set_num_threads = cxx.set_num_threads


@contextlib.contextmanager
def gperf_guard(filename: str):
    cxx.enable_gperf(filename)
    yield
    cxx.disable_gperf()
