
import sys
import importlib
from collections import OrderedDict


def print_versions():
    deps = ['numpy', 'scipy', 'pandas', 'dask', 'xarray', 'healpy']

    versions = OrderedDict()
    for dep in deps:
        try:
            if dep in sys.modules:
                mod = sys.modules[dep]
            else:
                mod = importlib.import_module(dep)
            version = mod.__version__
        except Exception:
            version = None
        versions[dep] = version

    for key in versions.keys():
        print('{}: {}'.format(key, versions[key]))
