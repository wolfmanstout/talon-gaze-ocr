__author__: str = "Max Bachmann"
__license__: str = "MIT"
__version__: str = "1.2.0"

def _fallback_import(module: str, name: str):
    import importlib
    import os

    impl = os.environ.get("JAROWINKLER_IMPLEMENTATION")

    if impl == "cpp":
        mod = importlib.import_module(module + "_cpp")
    elif impl == "python":
        mod = importlib.import_module(module + "_py")
    else:
        try:
            mod = importlib.import_module(module + "_cpp")
        except ModuleNotFoundError:
            mod = importlib.import_module(module + "_py")

    func = getattr(mod, name)
    if not func:
        raise ImportError(
            f"cannot import name '{name}' from '{mod.__name}' ({mod.__file__})"
        )
    return func

jaro_similarity = _fallback_import("jarowinkler._initialize", "jaro_similarity")
jarowinkler_similarity = _fallback_import("jarowinkler._initialize", "jarowinkler_similarity")