# Dimensionality Reduction Package
Python package for plug and play dimensionality reduction techniques, data clustering and visualization in a reduced space.
Using this package, you can reduce and plot according to a target variable your data set with a 3D o 2D chart and a matrix plot, without being worried about to normalize or scale your dataset for the different techniques. 

If you like the idea or you find usefull this repo in your job, please leave a ⭐ to support this personal project.

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

At the moment the package is not available using `pip install <PACKAGE-NAME>`.

For the installation from the source code click **[here](#installation)**.

Each available method returns a Pandas dataframe with number of components selected plus the target column; the first number of components minus one are the components obtained from the dimensionality reduction technique and the last column is the target variable passed as input.
Moreover the method creates two figures, the first one is a scatter plot (2D or 3D) of the reduced data points, and the second one is a pair plot. The 2D and the 3D plot is displayed only if the requested number of componenets are respectively 2 and 3.

## Summary table
Technique | Supervised Dataset | Unsupervised Dataset | Numerical Feature | Categorical Feature
--- | :---: | :---: | :---: | :---:
t-SNE | ✅ | ✅ | ✅ | ❌ 
LDA | ✅ | ❌ | ✅ | ❌ 
UMAP | ✅ | ✅ | ✅ | ❌
PCA | ✅ | ✅ | ✅ | ❌
FA | ✅ |✅ | ✅ | ❌
Truncated SVD | ✅ | ✅ | ✅ | ❌
Kernel PCA | ✅ | ✅ | ✅ | ❌
MDS | ✅ | ✅ | ✅ | ❌
Isomap | ✅ | ✅ | ✅ | ❌

> **Note**: since all the proposed techniques can't deal with categorical variable, it is possible to transform categorical variables into numerical one. I recommend to use the One Hot Encoding approach (`sklearn.preprocessing.OneHotEncoder`) when  the categorical variable takes on a large number of values (the unique value of the variable are few). For more information check out this [Kaggle article](https://www.kaggle.com/code/dansbecker/using-categorical-data-with-one-hot-encoding/notebook). 

## t-distributed Stochastic Neighbor Embedding (t-SNE)

### Description
It is a nonlinear unsupervised dimensionality reduction statistical method for visualizing high-dimensional data by giving each datanpoint a location in a low-dimensional space, typically two or three-dimensions. This technique finds clusters in data thereby making sure that an embedding preserves the meaning in the data so, t-SNE reduces dimensionality while trying to keep similar instances close and dissimilar instances apart.
This peculiarity leads to retain the major part of the original information on the final output. On the other hand, the parameterization of this method is not so easy and affects the final result, and therefore a good understanding of the parameters for t-SNE is necessary. 

In this package, the input parameters are default and for a first exploration of the data is quite good; for more fine tuning of the input parameters I recomend to read this article [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/).

t-SNE has been used for visualization in a wide range of applications, including genomics, computer security research, natural language processing, music analysis, cancer research, bioinformatics, geological domain interpretation, and biomedical signal processing [[Wikipedia](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)].

> **Note**: t-SNE output is just a projection into a lower dimensional space of the multi-dimensional input space. Thus, the output components are a mixture of the input features and the physical meaning and the measurement unit of the input feature is lost.

### References
1. Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11).
2. Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to use t-SNE effectively. Distill, 1(10), e2.

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
LDA is a method used in statistics and other fields, to find a linear combination of features that characterizes or separates two or more classes of objects or events. This dimensionality reduction is a supervised one, so you need to feed the algorithm with the feature matrix and the target column. The dimensionality of the result will be `n_classes - 1` [[Wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)].

LDA explicitly attempts to model the difference between the classes of data.

LDA works when the measurements made on independent variables for each observation are continuous quantities. When dealing with categorical independent variables, the equivalent technique is discriminant correspondence analysis, but at the moment this method is not implemented into the package.

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
UMAP is nonlinear unsupervised dimensionality reduction technique and it is an effective way for visualizing groups of data points and their relative proximities.
UMAP algorithm is competitive with t-SNE for visualization quality, and arguably preserves more of the global structure with superior run time performance [1].
To read me about the math behind this dimensionality reduction method I recommend this page: [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html).

### References
1. McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.

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
PCA is a statistical method, based on eigen-decomposition of the covariance matrix of the data. This process remaps the input data into a lower dimensional space, where each new dimension is a linear combination of the original features; the linear compination is performed in order to maximize the variance of the data in the new dimensional space.
The number of the principal components are the same of the features of the dataset, but each pricinpal component retain a certain amount of the original information; thus the first principal components retain the majority of the original information [[Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)].
PCA can be considered as a unsupervised linear dimensionality reduction technique.

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
Factor Analysis description is arriving!

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
Truncated SVD description is arriving!

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
Kernel PCA description is arriving!

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
Multidimensional Scaling description is arriving!

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
Isomap description is arriving!

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