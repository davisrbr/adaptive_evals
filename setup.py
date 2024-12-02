from setuptools import setup, find_packages

setup(
    name="adaptive_evals",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "inspect_ai",
        "numpy",
        "pandas",
        "datasets",
    ],
    python_requires=">=3.8",
    author="Davis Brown",
    author_email="davisrbr@seas.upenn.edu",
    description="Inspect AI attack experiments",
    keywords="ai, security, testing, adaptive evaluation",
) 