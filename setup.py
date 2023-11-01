from typing import List

from setuptools import find_packages
from setuptools import setup


def get_install_requires() -> List[str]:

    return [
        "PyYAML",
        # "ml_metrics",
        "optuna>=1.3.0",
        "lightgbm",
        "scikit-learn",
        "pyarrow",
    ]


def get_tests_require() -> List[str]:

    return ["pytest"]


setup(
    name="xfeat",
    version="0.1.1",
    description="Feature Engineering & Exploration Library using GPUs and Optuna",
    author="Kohei Ozaki",
    author_email="ozaki@preferred.jp",
    packages=find_packages(),
    install_requires=get_install_requires(),
    tests_require=get_tests_require(),
    setup_requires=["pytest-runner"],
    extras_require={"develop": ["pytest"]},
    entry_points={"console_scripts": []},
)
