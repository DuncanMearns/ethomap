import setuptools

setuptools.setup(
    name="ethomap",
    version="0.0.1",
    author="Duncan Mearns",
    author_email="duncan.mearns@bi.mpg.de",
    description="Generate behavior maps using non-linear embedding of dynamic time warping distances between time "
                "series.",
    packages=setuptools.find_packages(),
    python_requires=">=3.9"
)
