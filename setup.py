from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "zstd",
    "conway_tower",
    "numba",
    "numpy",
    "scikit-motionplan",
    "threadpoolctl",
    "ycb_utils",
    "plainmp>=0.3",
    "pr2_ikfast>=0.0.2",
]
private_requires = ["plainmp>=0.0.2"]


setup(
    name="rpbench",
    version="0.0.2",
    description="Robot planning benchmark",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    install_requires=install_requires,
    extras_require={"private": private_requires},
    packages=find_packages(exclude=("tests", "docs")),
    package_data={"rpbench": ["py.typed"]},
)
