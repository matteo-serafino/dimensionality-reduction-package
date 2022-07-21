from setuptools import setup, find_packages

with open("README.md") as readme:
    long_description = readme.read()

requirements = [
    "colorama>=0.4.5",
    "cycler>=0.11.0",
    "fonttools>=4.33.3",
    "joblib>=1.1.0",
    "kiwisolver>=1.4.3",
    "llvmlite>=0.38.1",
    "matplotlib>=3.5.2",
    "numba>=0.55.2",
    "numpy>=1.22.4",
    "packaging>=21.3",
    "pandas>=1.4.2",
    "Pillow>=9.1.1",
    "pynndescent>=0.5.7",
    "pyparsing>=3.0.9",
    "python-dateutil>=2.8.2",
    "pytz>=2022.1",
    "scikit-learn>=1.1.1",
    "scipy>=1.8.1",
    "seaborn>=0.11.2",
    "six>=1.16.0",
    "threadpoolctl>=3.1.0",
    "tqdm>=4.64.0",
    "umap-learn>=0.5.3"
]

setup(
    name='dimensionality-reduction-package',
    version='1.0.0',
    license='MIT',
    author="Matteo Serafino",
    author_email='matteo.serafino1@gmail.com',
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url='https://github.com/matteo-serafino/dimensionality-reduction-package',
    keywords='dimensionality reduction',
    install_requires=requirements,
    python_requires=">=3.6.2",
    include_package_data=True
)