from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="squarequant",
    version="0.1.0",
    author="Gabriel Bosch",
    author_email="contact@suqarequant.org",
    description="A Python package for quantitative finance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SquareQuant/squarequant-package",
    project_urls={
        "Documentation": "https://squarequant.readthedocs.io",
        "Bug Tracker": "https://github.com/SquareQuant/squarequant-package/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
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
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "optimization": ["cvxpy>=1.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    keywords="finance, risk, portfolio, investment, stocks, analysis, visualization",
)