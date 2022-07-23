import umap
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from dimensionality_reduction.services import plot_service as ps

TARGET_COLUMN = "target"

class DimensionalityReduction():

    def __init__(self):
        pass

    def tsne(self, X, y, n_components: int = 3, title: str = "t-SNE"):

        if n_components == 0:
            return None

        X_scaled = StandardScaler().fit(X).transform(X)
        
        columns = self.__columns_names(n_components)

        df = pd.DataFrame(TSNE(n_components=n_components, random_state=42).fit_transform(X_scaled), columns=columns)
        df[TARGET_COLUMN] = y

        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def UMAP(self, X, y, n_components: int = 3, title: str = "UMAP"):
        
        if n_components == 0:
            return None

        X_scaled = StandardScaler().fit(X).transform(X)

        columns = self.__columns_names(n_components)

        df = pd.DataFrame(umap.UMAP(random_state=42, n_components=n_components).fit_transform(X_scaled), columns=columns)
        df[TARGET_COLUMN] = y

        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def lda(self, X, y, title: str = "LDA"):

        n_components = len(np.unique(y)) - 1

        if n_components == 0:
            return None

        X_scaled = StandardScaler().fit(X).transform(X)

        columns = self.__columns_names(n_components)

        df = pd.DataFrame(LinearDiscriminantAnalysis(n_components=n_components).fit_transform(X_scaled, y), columns=columns)
        df[TARGET_COLUMN] = y

        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def pca(self, X, y, n_components: int = 3, title: str = "PCA"):

        if n_components == 0:
            return None

        X_scaled = StandardScaler().fit(X).transform(X)

        columns = self.__columns_names(n_components)

        df= pd.DataFrame(PCA(n_components=n_components).fit_transform(X_scaled), columns=columns)
        df[TARGET_COLUMN] = y

        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def factor_analysis(self, X, y, n_components: int = 3, title: str = "Factor Analysis"):

        if n_components == 0:
            return None

        X_scaled = StandardScaler().fit(X).transform(X)

        columns = self.__columns_names(n_components)

        df = pd.DataFrame(FactorAnalysis(n_components=n_components, rotation="varimax", random_state=42).fit_transform(X_scaled), columns=columns)
        df[TARGET_COLUMN] = y

        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def truncated_svd(self, X, y, n_components: int = 3, title: str = "Truncated SVD"):

        if n_components == 0:
            return None
        
        X_scaled = StandardScaler().fit(X).transform(X)

        columns = self.__columns_names(n_components)

        df = pd.DataFrame(TruncatedSVD(n_components=n_components, random_state=0).fit_transform(X_scaled), columns=columns)
        df[TARGET_COLUMN] = y
        
        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def kernel_pca(self, X, y, n_components: int = 3, title: str = "Kernel PCA"):

        if n_components == 0:
            return None

        X_scaled = StandardScaler().fit(X).transform(X)

        columns = self.__columns_names(n_components)

        df = pd.DataFrame(KernelPCA(n_components=n_components, kernel='rbf', gamma=15, random_state=42).fit_transform(X_scaled), columns=columns)
        df[TARGET_COLUMN] = y

        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def multidim_scaling(self, X, y, n_components: int = 3, title: str = "MDS"):

        if n_components == 0:
            return None

        X_scaled = StandardScaler().fit(X).transform(X)

        columns = self.__columns_names(n_components)

        df = pd.DataFrame(MDS(n_components=n_components, metric=True, random_state=42).fit_transform(X_scaled), columns=columns)
        df[TARGET_COLUMN] = y

        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def isomap(self, X, y, n_components: int = 3, title: str = "Isomap"):

        if n_components == 0:
            return None

        X_scaled = StandardScaler().fit(X).transform(X)

        columns = self.__columns_names(n_components)

        df = pd.DataFrame(Isomap(n_neighbors=10, n_components=n_components).fit_transform(X_scaled), columns=columns)
        df[TARGET_COLUMN] = y

        if n_components == 3:
            ps.plot_3d(df=df, title=title)

        if n_components == 2:
            ps.plot_2d(df=df, title=title)

        ps.matrix_plot(df=df)

        return df

    def __columns_names(self, n_components: int) -> list[str]:

        columns = []

        for i in range(n_components):
            columns.append(f"c_{i + 1}")
        
        return columns