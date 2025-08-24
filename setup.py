from setuptools import setup, Extension
from Cython.Build import cythonize

# List of Cython files to compile
cython_extensions = [
    Extension(
        name="qtradex.indicators.utilities",
        sources=["qtradex/indicators/utilities.pyx"],
    ),
    Extension(
        name="qtradex.indicators.qi",
        sources=["qtradex/indicators/qi.py"],
    ),
]

setup(
    name="QTradeX",
    version="1.1.0",
    setup_requires=["Cython>=0.29.21", "setuptools>=80"],
    description="AI-powered SDK featuring algorithmic trading, backtesting, deployment on 100+ exchanges, and multiple optimization engines.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    license="MIT",
    packages=["qtradex"],
    install_requires=[
        "ccxt",
        "jsonpickle",
        "setuptools>=64",
        "cachetools",
        "yfinance",
        "tulipy",
        "finance-datareader",
        "bitshares-signing",
        "numpy",
        "matplotlib",
        "scipy",
        "ttkbootstrap",
    ],
    entry_points={
        "console_scripts": [
            "qtradex-tune-manager=qtradex.core.tune_manager:main",
        ],
    },
    ext_modules=cythonize(cython_extensions, compiler_directives={'language_level': '3'}),
    url="https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK",
    project_urls={
        "Homepage": "https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK",
        "Issues": "https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK/issues",
    },
)
