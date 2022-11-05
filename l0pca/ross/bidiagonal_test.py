import abc
from l0pca.ross import bidiagonal_packed
from l0pca.ross import bidiagonal_reference
from l0pca.ross import dense_blocks
import numpy as np
import unittest

class BidiagonalTest(unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.packed_data = None
        self.temp_cell = None
        self.reference = bidiagonal_reference.ReferenceModule()
        self.packed = bidiagonal_packed.PackedModule()
    
    def test_3x3(self):
        self.packed_data = np.asarray(
            [[[0.75, 0.62, 0.15],
              [0., 0., 0.],
              [-0.52, 0.22, 0.]]],
            dtype=np.float32,
        )
        prod_data, prod_temp_cell = self.packed.update_augmented(
            self.packed_data, row_index=0)
        self.packed_data, self.temp_cell = self.reference.update_augmented(
            self.packed_data, row_index=0)
        # np.testing.assert_allclose(prod_data, self.packed_data)
        np.testing.assert_equal(prod_data.shape, self.packed_data.shape)

class AbstractBidiagonal(abc.ABC):

    def test_augmented(self):
        singular_values = np.asarray([0.57, -0.33], np.float32)
        augmented = np.asarray([-0.4, 0.15], np.float32)
        orthogonal_value = np.float32(0.42)
        packed = dense_blocks.pack_ross_matrix(singular_values, augmented, orthogonal_value)
        packed_result, temp_cell = self.module.update_augmented(packed, 0)
        # Orthogonal value has not changed.
        np.testing.assert_almost_equal(packed_result[0, 0, -1], orthogonal_value)
        self.assertEqual(temp_cell.row_index, 1)
        self.assertEqual(temp_cell.column_index, 0)
        np.testing.assert_almost_equal(temp_cell.value, [-0.5337], decimal=4)

class ReferenceTest(AbstractBidiagonal, unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.module = bidiagonal_reference.ReferenceModule()

class PackedTest(AbstractBidiagonal, unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.module = bidiagonal_packed.PackedModule()

    def test_prototype_9x9(self):
        values = dense_blocks.new_random_matrices(5, 9)
        bidiagonal_packed.diagonalize_packed_storage(values)

if __name__ == '__main__':
    unittest.main()