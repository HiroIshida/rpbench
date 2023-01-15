from setuptools import find_packages, setup

setup_requires = []

install_requires = ["numpy", "scikit-motionplan", "voxbloxpy", "threadpoolctl"]

setup(
    name="rpbench",
    version="0.0.0",
    description="Robot planning benchmark",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests", "docs")),
    package_data={"rpbench": ["py.typed"]},
)
