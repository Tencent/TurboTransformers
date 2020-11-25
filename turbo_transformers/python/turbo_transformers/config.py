try:
    from .turbo_transformers_cxx.config import *
except (ImportError, ModuleNotFoundError):  # debug build
    from .turbo_transformers_cxxd.config import *
