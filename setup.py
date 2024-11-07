import os

from setuptools import find_packages, setup

version = {}
with open(os.path.join("s2mosaic", "__version__.py")) as fp:
    exec(fp.read(), version)

setup(
    name="s2mosaic",
    version=version["__version__"],
    description="""Python library for making cloud-free Sentinel-2 mosaics""",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nick Wright",
    author_email="nicholas.wright@dpird.wa.gov.au",
    url="https://github.com/DPIRD-DMA/S2Mosaic",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "planetary_computer",
        "pystac_client",
        "geopandas",
        "omnicloudmask>=1.0.3",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_data={"s2mosaic": ["S2_grid/sentinel_2_index.gpkg"]},
)
