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
    use_scm_version=True,
    setup_requires=["Cython", "setuptools_scm", "setuptools>80"],
    description="AI-powered SDK featuring algorithmic trading, backtesting, deployment on 100+ exchanges, and multiple optimization engines.",
    long_description=open("README.md").read(),
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
    url="https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK",
    project_urls={
        "Homepage": "https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK",
        "Issues": "https://github.com/squidKid-deluxe/QTradeX-Algo-Trading-SDK/issues",
    },
)
