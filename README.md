# Dimensionality Reduction Package
Python package for plug and play dimensionality reduction techniques, data clustering and visualization in a reduced space.
Using this package, you can reduce and plot according a target variable your data set with a 3D o 2D chart and a matrix plot.

The available techniques are:
* [t-distributed Stochastic Neighbor Embedding (t-SNE)](#t-distributed-stochastic-neighbor-embedding-t-sne);
* [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda);
* [Uniform Manifold Approximation and Projection (UMAP)](#uniform-manifold-approximation-and-projection-umap);
* [Principal Component Analysis (PCA)](#principal-component-analysis-pca);
* [Factor Analysis (FA)](#factor-analysis-fa);
* [Truncated Singular Value Decompisition (SVD)](#truncated-singular-value-decomposition-svd);
* [Kernel Principal Component Analysis](#kernel-principal-component-analysis-pca);
* [Multidimensional Scaling (MDS)](#multidimensional-scaling);
* [Isometric Mapping (Isomap)](#isometric-mapping-isomap).

At the moment the packege is not available using `pip install <PACKAGE-NAME>`.
For the installation from the source code click [here](#installation).

Each available method returns a pandas dataframe with number of components selected plus the target column; the first number of components minus one are the componenets obtained from the dimensionality reduction technique and the last one is the target variable passed as input.
Moreover the method creates two figures, the first one is a scatter plot (2D or 3D) of the reducted data points, and the second one is a pair plot. The 2D and the 3D plot is displayed only if the requested number of componenets are respectively 2 and 3.

## t-distributed Stochastic Neighbor Embedding (t-SNE)

### Description
t-SNE description goes here.

### Use Cases
t-SNE use cases goes here.

### Examples
 ```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.tsne(X, y, n_components=2)
```


## Linear Discriminant Analysis (LDA)

### Description
LDA description goes here.

### Use Cases
LDA use cases goes here.

### Examples
 ```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.lda(X, y)
```


## Uniform Manifold Approximation and Projection (UMAP)

### Description
UMAP description goes here.

### Use Cases
UMAP use cases goes here.

### Examples
 ```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.umap(X, y, n_components=2)
```


## Principal Component Analysis (PCA)

### Description
PCA description goes here.

### Use Cases
PCA use cases goes here.

### Examples
```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.pca(X, y, n_components=2)
```


## Factor Analysis (FA)

### Description
Factor Analysis description goes here.

### Use Cases
Factor Analysis use cases goes here.

### Examples
```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.factor_analysis(X, y, n_components=2)
```


## Truncated Singular Value Decomposition (SVD)

### Description
Truncated SVD description goes here.

### Use Cases
Truncated SVD use cases goes here.

### Examples
```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.truncated_svd(X, y, n_components=2)
```


## Kernel Principal Component Analysis (PCA)

### Description
Kernel PCA description goes here.

### Use Cases
Kernel PCA use cases goes here.

### Examples
```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.kernel_pca(X, y, n_components=2)
```


## Multidimensional Scaling

### Description
Multidimensional Scaling description goes here.

### Use Cases
Multidimensional Scaling use cases goes here.

### Examples
```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.multidim_scaling(X, y, n_components=2)
```


## Isometric Mapping (Isomap)

### Description
Isomap description goes here.

### Use Cases
Isomap use cases goes here.

### Examples
```python
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction
from sklearn import datasets

dr = DimensionalityReduction()

iris_dataset = datasets.load_iris()

X = iris_dataset.data[:, :3]
y = iris_dataset.target

df = dr.isomap(X, y, n_components=2)
```


## Installation
For the installation from the source code type this command into your terminal window:
```
pip install git+<repository-link>
```
or
```
python -m pip install git+<repository-link>
```
or
```
python3 -m pip install git+<repository-link>
```