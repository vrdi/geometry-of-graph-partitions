from setuptools import find_packages, setup

with open("./README.rst") as f:
    long_description = f.read()

setup(
    name="wasserplan",
    description="1-Wasserstein distances between districting plans",
    author="Metric Geometry and Gerrymandering Group",
    author_email="gerrymandr@gmail.com",
    maintainer="Parker J. Rule",
    maintainer_email="parker.rule@tufts.edu",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/vrdi/geometry-of-graph-partitions",
    packages=['wasserplan'],
    version='0.1.3',
    install_requires=['gerrychain', 'cvxpy'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ]
)
