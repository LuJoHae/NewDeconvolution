import pkgutil
import importlib
import pathlib

# get directory of this package
pkg_dir = pathlib.Path(__file__).parent

for module in pkgutil.iter_modules([str(pkg_dir)]):
    if module.name.startswith("_"):
        continue  # skip private modules

    importlib.import_module(f"{__name__}.{module.name}")