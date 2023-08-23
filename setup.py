import setuptools

with open("README.md", "r") as descr:
    long_description = descr.read()

setuptools.setup(
    name="Projected exponenial methods",
    version="1.0",
    author="Benjamin Carrel",
    author_email="benjamin.carrel@unige.ch",
    url="https://gitlab.unige.ch/Benjamin.Carrel/projected-exponential-methods",
    description="Implementation of the projected exponential methods for solving differential Sylvester-like equations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "jupyter"
    ]
)