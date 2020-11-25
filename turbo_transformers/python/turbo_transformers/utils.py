try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
import contextlib

__all__ = [
    'pref_guard', 'set_num_threads', 'set_stderr_verbose_level',
    'disable_perf', 'enable_perf', 'reset_allocator_schema',
    'bert_opt_mem_allocate_api'
]

set_num_threads = cxx.set_num_threads
set_stderr_verbose_level = cxx.set_stderr_verbose_level

disable_perf = cxx.disable_perf
enable_perf = cxx.enable_perf
reset_allocator_schema = cxx.reset_allocator_schema
bert_opt_mem_allocate_api = cxx.bert_opt_mem_allocate_api


@contextlib.contextmanager
def pref_guard(filename: str):
    cxx.enable_perf(filename)
    yield
    cxx.disable_perf()
