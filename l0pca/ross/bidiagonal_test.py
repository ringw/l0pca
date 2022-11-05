import abc
from l0pca.ross import bidiagonal_packed
from l0pca.ross import bidiagonal_reference
from l0pca.ross import dense_blocks
import numpy as np
import unittest

class AbstractBidiagonal(abc.ABC):

    def test_augmented(self):
        singular_values = np.asarray([0.57, -0.33], np.float32)
        augmented = np.asarray([-0.4, 0.15], np.float32)
        orthogonal_value = np.float32(0.42)
        packed = dense_blocks.pack_ross_matrix(singular_values, augmented, orthogonal_value)
        packed_result, temp_cell = self.module.update_augmented(packed, 0)
        # Orthogonal value has not changed.
        np.testing.assert_almost_equal(packed_result[0, 0, -1], orthogonal_value)
        self.assertEqual(temp_cell.row_index, -1)
        self.assertEqual(temp_cell.column_index, -1)
        self.assertEqual(temp_cell.value, 0.)

class ReferenceTest(AbstractBidiagonal, unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.module = bidiagonal_reference.ReferenceModule()

class PackedTest(AbstractBidiagonal, unittest.TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.module = bidiagonal_packed.PackedModule()

if __name__ == '__main__':
    unittest.main()