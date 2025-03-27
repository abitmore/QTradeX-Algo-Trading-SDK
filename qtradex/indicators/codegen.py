import importlib
import re

MODULES = ["tulipy"]

REPLACE = r"(\w+)"
SUB = r"""
@cache
def \1(*args, **kwargs):
    return tulipy.\1(*args, **kwargs)
"""


def main():
    imports = ""
    for module in MODULES:
        print(f"Generating cache wrapper for {module}...")
        # only take lowercase functions that do not start with a dunderscore
        functions = [
            i
            for i in dir(importlib.import_module(module))
            if not any([i.startswith("__"), i.lower() != i])
        ]
        functions = "\n".join(functions)
        code = re.sub(REPLACE, SUB, functions)
        code = (
            f"from qtradex.indicators.cache_decorator import cache\nimport {module}\n\n"
            + code
        )
        with open(f"{module}_wrapped.py", "w") as handle:
            handle.write(code)
            handle.close()
        imports += f"import qtradex.indicators.{module}_wrapped as {module}\n"
    imports += "from qtradex.indicators.utilities import derivative, float_period, lag\n"
    with open(f"__init__.py", "w") as handle:
        handle.write(imports)
        handle.close()


if __name__ == "__main__":
    main()
