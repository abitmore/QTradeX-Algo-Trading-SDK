import importlib
import inspect
import re

MODULES = ["tulipy"]

SUB = """
@cache
@float_period{periods}
def {name}(*args, **kwargs):
    return tulipy.{name}(*args, **kwargs)
"""


def main():
    imports = ""
    for module in MODULES:
        print(f"Generating cache wrapper for {module}...")
        # only take lowercase functions that do not start with a dunderscore
        mod_obj = importlib.import_module(module)
        functions = [
            i
            for i in dir(mod_obj)
            if not any([i.startswith("__"), i.lower() != i])
            and callable(getattr(mod_obj, i))
        ]
        parameters = [
            inspect.signature(getattr(mod_obj, func)).parameters for func in functions
        ]
        code = "\n".join(i.replace("@float_period()\n", "") for i in [
            SUB.format(
                name=func,
                periods=tuple(
                    idx for idx, i in enumerate(para.keys()) if i.endswith("period")
                ),
            )
            for func, para in zip(functions, parameters)
        ])
        code = (
            f"from qtradex.indicators.cache_decorator import cache, float_period\nimport {module}\n\n"
            + code
        )
        with open(f"{module}_wrapped.py", "w") as handle:
            handle.write(code)
            handle.close()
        imports += f"import qtradex.indicators.{module}_wrapped as {module}\n"
    imports += (
        "from qtradex.indicators.utilities import derivative, float_period, lag\nimport qtradex.indicators.fitness\n"
    )
    with open(f"__init__.py", "w") as handle:
        handle.write(imports)
        handle.close()


if __name__ == "__main__":
    main()
