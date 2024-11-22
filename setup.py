from setuptools import setup, find_packages

setup(
    name="inspect_attacks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "inspect_ai",  # Add other dependencies here
        "numpy",
    ],
    python_requires=">=3.8",
    author="Davis Brown",
    author_email="davisrbr@seas.upenn.edu",
    description="Inspect AI attack experiments",
    keywords="ai, security, testing",
) 