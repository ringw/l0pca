import abc
import dataclasses
import numpy as np

@dataclasses.dataclass
class TempCell:
    value: np.float32
    row_index: int
    column_index: int

class BidiagonalModule(abc.ABC):

    def bidiagonalize_index(self, sparse_packed, row_index):
        sparse_packed, temp_cell = self.update_augmented(sparse_packed, row_index)
        for i in range(row_index - 1, 0, -1):
            sparse_packed, temp_cell = self.move_temp_cell(sparse_packed, temp_cell)
        return sparse_packed. temp_cell

    @abc.abstractmethod
    def update_augmented(sparse_packed, row_index):
        pass

    @abc.abstractmethod
    def move_temp_cell(sparse_packed, temp_cell):
        pass
