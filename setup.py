from setuptools import setup

# These are here for GitHub's dependency graph.
setup(
    name="Probability theory lab 01",
    install_requires=[
        "PyQt5 >= 5.15.4",
        "matplotlib >= 3.4.3",
        "numba >= 0.53.1",
        "numpy >= 1.21.1",
        "scipy >= 1.7.1",
    ],
)
