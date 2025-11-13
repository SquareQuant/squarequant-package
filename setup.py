from setuptools import setup, find_packages
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="squarequant",
    version="0.2.3",
    author="Gabriel Bosch",
    author_email="contact@squarequant.org",
    description="A Python package for financial risk metrics and stock data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SquareQuant/squarequant-package",
    project_urls={
        "Documentation": "https://www.squarequant.org",
        "Bug Tracker": "https://github.com/SquareQuant/squarequant-package/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "yfinance>=0.1.63",
        "scipy>=1.5.0",
        "scikit-learn>=1.5.0",
        "matplotlib>=3.3.0",
        "statsmodels>=0.12.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "optimization": ["cvxpy>=1.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.22.0",
            "myst-parser>=1.0.0",
            "pytest-cov>=2.12.0",
        ],
    },
    setup_requires=["pandas", "numpy"],
    include_dirs=[numpy.get_include()],
    keywords="finance, risk, portfolio, investment, stocks, analysis, drawdown, var, cvar, entropic, ulcer index, theta data, quant, time series, monte carlo",
)
