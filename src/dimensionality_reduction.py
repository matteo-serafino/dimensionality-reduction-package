import imp
import umap
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from plot import Plot

class DimensionalityReduction():

    def __init__(self):
        pass

    def tsne(self, X, y, n_components:int = 3):
         
        X_scaled = StandardScaler().fit(X).transform(X)

        if (n_components == 3):

            print("t-SNE 3d dimensionality reduction running...")
            df_tsne = pd.DataFrame(TSNE(n_components=n_components).fit_transform(X_scaled), columns=["component_1", "component_2", "component_3"])
            df_tsne["label"] = y
            Plot.plot_3d(df=df_tsne, title="t-SNE")

        elif (n_components == 2):

            print("t-SNE 2d dimensionality reduction running...")
            df_tsne = pd.DataFrame(TSNE(n_components=n_components).fit_transform(X_scaled), columns=["component_1", "component_2"])
            df_tsne["label"] = y
            Plot.plot_2d(df=df_tsne, title="t-SNE")

        elif (n_components == 1):

            print("t-SNE 1d dimensionality reduction running...")
            df_tsne = pd.DataFrame(TSNE(n_components=n_components).fit_transform(X_scaled), columns=["component_1"])
            df_tsne["label"] = y

        else:
            return None

        Plot.matrix_plot(df=df_tsne)

        return df_tsne

    def UMAP(self, X, y, n_components:int = 3):

        X_scaled = StandardScaler().fit(X).transform(X)

        if (n_components == 3):

            print("UMAP 3d dimensionality reduction running...")
            df_umap = pd.DataFrame(umap.UMAP(random_state=42, n_components=n_components).fit_transform(X_scaled), columns=["component_1", "component_2", "component_3"])
            df_umap["label"] = y
            Plot.plot_3d(df=df_umap, title="UMAP")

        elif (n_components == 2):

            print("UMAP 2d dimensionality reduction running...")
            df_umap = pd.DataFrame(umap.UMAP(random_state=42, n_components=n_components).fit_transform(X_scaled), columns=["component_1", "component_2"])
            df_umap["label"] = y
            Plot.plot_2d(df=df_umap, title="UMAP")

        elif (n_components == 1):

            print("UMAP 1d dimensionality reduction running...")
            df_umap = pd.DataFrame(umap.UMAP(random_state=42, n_components=n_components).fit_transform(X_scaled), columns=["component_1"])
            df_umap["label"] = y
        
        else:
            return None

        Plot.matrix_plot(df=df_umap)

        return df_umap

    def lda(self, X, y):

        X_scaled = StandardScaler().fit(X).transform(X)
        n_classes = len(y.unique())

        if (n_classes == 2):

            print("LDA 1d dimensionality reduction running...")
            df_lda = pd.DataFrame(LinearDiscriminantAnalysis(n_components=1).fit_transform(X_scaled, y), columns=["component_1"])
            df_lda["label"] = y

        elif (n_classes == 3):

            print("LDA 2d dimensionality reduction running...")
            df_lda = pd.DataFrame(LinearDiscriminantAnalysis(n_components=2).fit_transform(X_scaled, y), columns=["component_1", "component_2"])
            df_lda["label"] = y
            Plot.plot_2d(df=df_lda, title="LDA")

        elif (n_classes == 4):

            print("LDA 3d dimensionality reduction running...")
            df_lda = pd.DataFrame(LinearDiscriminantAnalysis(n_components=3).fit_transform(X_scaled, y), columns=["component_1", "component_2", "component_3"])
            df_lda["label"] = y
            Plot.plot_3d(df=df_lda, title="LDA")
        
        else:
            return None

        Plot.matrix_plot(df=df_lda)

        return df_lda

    def pca(self, X, y, n_components:int = 3):

        X_scaled = StandardScaler().fit(X).transform(X)

        if (n_components == 3):

            print("PCA 3d dimensionality reduction running...")
            df_pca = pd.DataFrame(PCA(n_components=n_components).fit_transform(X_scaled), columns=["component_1", "component_2", "component_3"])
            df_pca["label"] = y
            Plot.plot_3d(df=df_pca, title="PCA")

        elif(n_components == 2):

            print("PCA 2d dimensionality reduction running...")
            df_pca = pd.DataFrame(PCA(n_components=n_components).fit_transform(X_scaled), columns=["component_1", "component_2"])
            df_pca["label"] = y
            Plot.plot_2d(df=df_pca, title="PCA")

        elif(n_components == 1):

            print("PCA 1d dimensionality reduction computing...")
            df_pca = pd.DataFrame(PCA(n_components=n_components).fit_transform(X_scaled), columns=["component_1"])
            df_pca["label"] = y

        else:
            return None

        Plot.matrix_plot(df=df_pca)

        return df_pca

    def factor_analysis(self, X, y, n_components:int = 3):

        X_scaled = StandardScaler().fit(X).transform(X)

        if (n_components == 3):

            print("Factor Analysis 3d dimensionality reduction running...")
            df_fa = pd.DataFrame(FactorAnalysis(n_components=n_components, rotation="varimax", random_state=0).fit_transform(X_scaled), 
                columns=["component_1", "component_2", "component_3"])
            df_fa["label"] = y
            Plot.plot_3d(df=df_fa, title="Factor Analysis")

        elif(n_components == 2):

            print("Factor Analysis 2d dimensionality reduction running...")
            df_fa = pd.DataFrame(FactorAnalysis(n_components=n_components, rotation="varimax", random_state=0).fit_transform(X_scaled),
                columns=["component_1", "component_2"])
            df_fa["label"] = y
            Plot.plot_2d(df=df_fa, title="Factor Analysis")

        elif(n_components == 1):

            print("Factor Analysis 1d dimensionality reduction computing...")
            df_fa = pd.DataFrame(FactorAnalysis(n_components=n_components, rotation="varimax", random_state=0).fit_transform(X_scaled),
                columns=["component_1"])
            df_fa["label"] = y

        else:
            return None

        Plot.matrix_plot(df=df_fa)

        return df_fa

    def truncated_SVD(self, X, y, n_components:int = 3):

        X_scaled = StandardScaler().fit(X).transform(X)

        if (n_components == 3):
            print("Truncated SVD 3d dimensionality reduction running...")
            df_tsvd = pd.DataFrame(TruncatedSVD(n_components=n_components, random_state=0).fit_transform(X_scaled), 
                columns=["component_1", "component_2", "component_3"])
            df_tsvd["label"] = y
            Plot.plot_3d(df=df_tsvd, title="Truncated SVD")

        elif(n_components == 2):

            print("Truncated SVD 2d dimensionality reduction running...")
            df_tsvd = pd.DataFrame(TruncatedSVD(n_components=n_components, random_state=0).fit_transform(X_scaled),
                columns=["component_1", "component_2"])
            df_tsvd["label"] = y
            Plot.plot_2d(df=df_tsvd, title="Truncated SVD")

        elif(n_components == 1):

            print("Truncated SVD 1d dimensionality reduction computing...")
            df_tsdv = pd.DataFrame(TruncatedSVD(n_components=n_components, random_state=0).fit_transform(X_scaled),
                columns=["component_1"])
            df_tsdv["label"] = y

        else:
            return None

        Plot.matrix_plot(df=df_tsvd)

        return df_tsvd

    def kernel_pca(self, X, y, n_components:int):

        X_scaled = StandardScaler().fit(X).transform(X)

        if (n_components == 3):
            print("Kernel PCA 3d dimensionality reduction running...")
            df_kpca = pd.DataFrame(KernelPCA(n_components=n_components, kernel='rbf', gamma=15, random_state=42).fit_transform(X_scaled), 
                columns=["component_1", "component_2", "component_3"])
            df_kpca["label"] = y
            Plot.plot_3d(df=df_kpca, title="Kernel PCA")

        elif(n_components == 2):

            print("Kernel PCA 2d dimensionality reduction running...")
            df_kpca = pd.DataFrame(KernelPCA(n_components=n_components, kernel='rbf', gamma=15, random_state=42).fit_transform(X_scaled),
                columns=["component_1", "component_2"])
            df_kpca["label"] = y
            Plot.plot_2d(df=df_kpca, title="Kernel PCA")

        elif(n_components == 1):

            print("Kernel PCA 1d dimensionality reduction computing...")
            df_kpca = pd.DataFrame(KernelPCA(n_components=n_components, kernel='rbf', gamma=15, random_state=42).fit_transform(X_scaled),
                columns=["component_1"])
            df_kpca["label"] = y

        else:
            return None

        Plot.matrix_plot(df=df_kpca)

        return df_kpca

    def multidim_scaling(self, X, y, n_components:int):

        X_scaled = StandardScaler().fit(X).transform(X)

        if (n_components == 3):
            print("MDS 3d dimensionality reduction running...")
            df_mds = pd.DataFrame(MDS(n_components=n_components, metric=True, random_state=42).fit_transform(X_scaled), 
                columns=["component_1", "component_2", "component_3"])
            df_mds["label"] = y
            Plot.plot_3d(df=df_mds, title="MDS")

        elif(n_components == 2):

            print("MDS 2d dimensionality reduction running...")
            df_mds = pd.DataFrame(MDS(n_components=n_components, metric=True, random_state=42).fit_transform(X_scaled),
                columns=["component_1", "component_2"])
            df_mds["label"] = y
            Plot.plot_2d(df=df_mds, title="MDS")

        elif(n_components == 1):

            print("MDS 1d dimensionality reduction computing...")
            df_mds = pd.DataFrame(MDS(n_components=n_components, metric=True, random_state=42).fit_transform(X_scaled),
                columns=["component_1"])
            df_mds["label"] = y

        else:
            return None

        Plot.matrix_plot(df=df_mds)

        return df_mds

    def isomap(self, X, y, n_components:int):

        X_scaled = StandardScaler().fit(X).transform(X)

        if (n_components == 3):
            print("Isomap 3d dimensionality reduction running...")
            df_isomap = pd.DataFrame(Isomap(n_neighbord=10, n_components=n_components).fit_transform(X_scaled), 
                columns=["component_1", "component_2", "component_3"])
            df_isomap["label"] = y
            Plot.plot_3d(df=df_isomap, title="Isomap")

        elif(n_components == 2):

            print("Isomap 2d dimensionality reduction running...")
            df_isomap = pd.DataFrame(Isomap(n_neighbord=10, n_components=n_components).fit_transform(X_scaled),
                columns=["component_1", "component_2"])
            df_isomap["label"] = y
            Plot.plot_2d(df=df_isomap, title="Isomap")

        elif(n_components == 1):

            print("Isomap 1d dimensionality reduction computing...")
            df_isomap = pd.DataFrame(Isomap(n_neighbord=10, n_components=n_components).fit_transform(X_scaled),
                columns=["component_1"])
            df_isomap["label"] = y

        else:
            return None

        Plot.matrix_plot(df=df_isomap)

        return df_isomap