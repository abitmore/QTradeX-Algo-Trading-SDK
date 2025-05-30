from Cython.Build import cythonize
from setuptools import Extension, setup

# List of Cython files to compile
cython_extensions = [
    Extension(
        name="qtradex.indicators.utilities",
        sources=["qtradex/indicators/utilities.pyx"],
    ),
    Extension(name="qtradex.indicators.qi", sources=["qtradex/indicators/qi.py"]),
]

setup(
    name="QTradeX",
    version="1.0",
    description="",  # Add your description here
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    license="MIT",
    packages=["qtradex"],  # Adjust if you have additional packages
    install_requires=[
        "ccxt",
        "jsonpickle",
        "Cython",
        "setuptools>64",
        "cachetools",
        "yfinance",
        "tulipy",
        "finance-datareader",
        "bitshares-signing",
        "numpy",
        "matplotlib",
        "scipy",
        "Cython",
        "ttkbootstrap",
    ],
    entry_points={
        "console_scripts": [
            "qtradex-tune-manager=qtradex.core.tune_manager:main",
        ],
    },
    ext_modules=cythonize(cython_extensions),
    url="https://github.com/",
    project_urls={
        "Homepage": "https://github.com/",
        "Issues": "https://github.com/",
    },
)
