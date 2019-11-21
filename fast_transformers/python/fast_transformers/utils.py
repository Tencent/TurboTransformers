import fast_transformers.fast_transformers_cxx as cxx
import contextlib

__all__ = ['gperf_guard']


@contextlib.contextmanager
def gperf_guard(filename: str):
    cxx.enable_gperf(filename)
    yield
    cxx.disable_gperf()
