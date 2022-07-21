import unittest
from sklearn import datasets
from dimensionality_reduction.dimensionality_reduction import DimensionalityReduction

dr = DimensionalityReduction()

class TestDimensionalityReduction(unittest.TestCase):

    def setUp(self):

        iris_dataset = datasets.load_iris()

        self.X = iris_dataset.data[:, :3]
        self.y = iris_dataset.target

    def test_tsne(self):

        df = dr.tsne(self.X, self.y, 2)

        assert df.shape == (150, 3)

    def test_umap(self):

        df = dr.UMAP(self.X, self.y, 2)

        assert df.shape == (150, 3)

    def test_lda(self):

        df = dr.lda(self.X, self.y)

        assert df.shape == (150, 3)   

    def test_pca(self):

        df = dr.pca(self.X, self.y, 2)

        assert df.shape == (150, 3)

    def test_factor_analysis(self):

        df = dr.factor_analysis(self.X, self.y, 2)

        assert df.shape == (150, 3)

    def test_truncated_svd(self):

        df = dr.truncated_svd(self.X, self.y, 2)

        assert df.shape == (150, 3)

    def test_kernal_pca(self):

        df = dr.kernel_pca(self.X, self.y, 2)

        assert df.shape == (150, 3)

    def test_multidim_scaling(self):

        df = dr.multidim_scaling(self.X, self.y, 2)

        assert df.shape == (150, 3)

    def test_isomap(self):

        df = dr.isomap(self.X, self.y, 2)

        assert df.shape == (150, 3)