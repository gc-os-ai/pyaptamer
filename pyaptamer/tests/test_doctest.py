"""Doctest checks directed through pytest with conditional skipping."""
# copied from skbase/sktime under BSD-3-Clause License

import importlib
import inspect
import pkgutil
from functools import lru_cache


EXCLUDE_MODULES_STARTING_WITH = ("all", "test")


def _all_objects(module_name):
    """Get all objects from a module, including submodules.

    Excludes modules starting with 'all' or 'test'.

    Parameters
    ----------
    module_name : str
        Name of the module.

    Returns
    -------
    obj_list : list
        List of tuples (obj_name, object).
    """
    res = _all_objects_cached(module_name)
    # copy the result to avoid modifying the cached result
    return res.copy()


@lru_cache
def _all_objects_cached(module_name):
    """Get all objects from a module, including submodules.

    Excludes modules starting with 'all' or 'test'.

    Parameters
    ----------
    module_name : str
        Name of the module.

    Returns
    -------
    obj_list : list
        List of tuples (obj_name, object).
    """
    # Import the package
    package = importlib.import_module(module_name)

    # Initialize an empty list to hold all objects
    obj_list = []

    # Walk through the package's modules
    package_path = package.__path__[0]
    for _, modname, _ in pkgutil.walk_packages(
        path=[package_path], prefix=package.__name__ + "."
    ):
        # Skip modules starting with 'all' or 'test'
        if modname.split(".")[-1].startswith(EXCLUDE_MODULES_STARTING_WITH):
            continue

        # Import the module
        module = importlib.import_module(modname)

        # Get all functions from the module
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # if function is imported from another module, skip it
            if obj.__module__ != module.__name__:
                continue
            # add the function to the list
            obj_list.append((name, obj))

        # Get all classes from the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # if class is imported from another module, skip it
            if obj.__module__ != module.__name__:
                continue
            # add the function to the list
            obj_list.append((name, obj))

    return obj_list


def pytest_generate_tests(metafunc):
    """Test parameterization routine for pytest.

    Fixtures parameterized
    ----------------------
    func : all functions and classes from the package
    """
    # we assume all four arguments are present in the test below
    objs_and_names = _all_objects("pyaptamer")

    if len(objs_and_names) > 0:
        names, objs = zip(*objs_and_names)

        metafunc.parametrize("obj", objs, ids=names)
    else:
        metafunc.parametrize("obj", [])


def test_all_functions_doctest(obj):
    """Run doctest for all objects in the package."""
    from skbase.utils.doctest_run import run_doctest

    run_doctest(obj, name=f"{obj.__name__}")
