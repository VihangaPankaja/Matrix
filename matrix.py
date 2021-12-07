from copy import deepcopy
from functools import cached_property
from itertools import chain
from decimal import Decimal
from typing import Iterable, Union

__all__ = ['Matrix']
__author__ = "Vihanga_Pankaja"
__version__ = '0.1.0'


class MatrixError(Exception):
    """
    Exception class for errors in Matrix
    """

    def __init__(self, *args):
        super().__init__(*args)

    class ElementError(Exception):
        def __init__(self):
            self.message = "Matrix must have equal number of elements in every row"
            super().__init__(self.message)

    class UnsupportedElement(Exception):
        def __init__(self, type, row, col):
            self.message = f"Unsupported element type ({type}) detected at ({row},{col})"
            super().__init__(self.message)

    class NotSquareMatrix(Exception):
        def __init__(self):
            self.message = "Not a square matrix"
            super().__init__(self.message)

    class UnsupportedType(Exception):
        def __init__(self):
            self.message = "Unsupported type (only int, float, complex are supported)"
            super().__init__(self.message)

    class NotSameOrder(Exception):
        def __init__(self):
            self.message = "Two matrices are not in the same order"
            super().__init__(self.message)

    class CannotMultiply(Exception):
        def __init__(self):
            self.message = "Number of columns of the first matrix should be equal to the number of rows of the second matrix"
            super().__init__(self.message)

    class NotSameElementCount(Exception):
        def __init__(self, type=None):
            self.message = "Need to contain same number of elements in the reshaped matrix"
            if type is not None:
                self.message += ". getting not fully filled row or column to given attributes"
            super().__init__(self.message)

    class ElementTypeError(Exception):
        def __init__(self):
            self.message = "only int objects are allowed with bitwise operations"
            super().__init__(self.message)


class Matrix:   # TODO: exception handling not done completely

    def __init__(self, elements: Iterable[Iterable[Union[int, float, complex]]]) -> None:
        """
        Parameters
        ----------
        elements : Iterable[Iterable[Union[int | float | complex]]]
            any 2 dimensional iterable with int, float, complex elements that have equal elements in each row

        Raises
        ------
        MatrixError.ElementError
            when not equal number of elements presents in each row
        MatrixError.UnsupportedElement
            when elements contain different type of objects other than integers, floats ans complex numbers
        """

        if not all(len(elements[0]) == len(x) for x in elements[1:]):
            raise MatrixError.ElementError
        for row_num, row in enumerate(elements):
            for col_num, element in enumerate(row):
                # elements must be int, float or complex
                if not any(map(isinstance, [element]*3, [int, float, complex])):
                    raise MatrixError.UnsupportedElement(
                        type(element), row_num+1, col_num+1)

        self._matrix = list(map(list, elements))
        self.ElementaryTransform = __class__.__ElementaryTransform(self)
        self.LinearTransform = __class__.__LinearTransform(self)

    @cached_property
    def order(self) -> tuple[int, int]:
        return len(self._matrix), len(self._matrix[0])

    @cached_property
    def is_square_matrix(self) -> bool:
        return self.order[0] == self.order[1]

    @property
    def is_singular_matrix(self) -> bool:
        return abs(self) == 0

    @property
    def is_row_matrix(self) -> bool:
        return self.order[0] == 1

    @property
    def is_column_matrix(self) -> bool:
        return self.order[1] == 1

    @property
    def is_orthogonal_matrix(self) -> bool:
        return self.is_square_matrix and __class__.is_identical(self.transpose, self.inverse)

    @property
    def is_triangular_matrix(self) -> bool:
        return self.is_upper_triangular_matrix or self.is_lower_triangular_matrix

    @property
    def is_upper_triangular_matrix(self) -> bool:
        if not self.is_square_matrix:
            raise MatrixError.NotSquareMatrix

        return (not any(self._matrix[i][j] for i in range(self.order[0]) for j in range(self.order[1]) if i > j)
                and any(self._matrix[i][j] for i in range(self.order[0]) for j in range(self.order[1]) if i <= j))

    @property
    def is_lower_triangular_matrix(self) -> bool:
        if not self.is_square_matrix:
            raise MatrixError.NotSquareMatrix

        return (not any(self._matrix[i][j] for i in range(self.order[0]) for j in range(self.order[1]) if i < j)
                and any(self._matrix[i][j] for i in range(self.order[0]) for j in range(self.order[1]) if i >= j))

    @property
    def is_diagonal_matrix(self) -> bool:
        if not self.is_square_matrix:
            raise MatrixError.NotSquareMatrix

        return (not any(self._matrix[i][j] for i in range(self.order[0]) for j in range(self.order[1]) if i != j)
                and any(self._matrix[i][i] for i in range(self.order[0])))

    @property
    def identity(self) -> 'Matrix':
        if not self.is_square_matrix:
            raise MatrixError.NotSquareMatrix

        tmp = list([0] * self.order[0] for _ in range(self.order[0]))
        for i in range(self.order[0]):
            tmp[i][i] = 1

        return __class__(tmp)

    @property
    def transpose(self) -> 'Matrix':
        tmp = list([0]*self.order[0] for _ in range(self.order[1]))

        for i in range(self.order[0]):
            for j in range(self.order[1]):
                tmp[j][i] = self._matrix[i][j]

        return __class__(tmp)

    @property
    def adjoint(self) -> 'Matrix':
        if not self.is_square_matrix:
            raise MatrixError.NotSquareMatrix

        def cofactor(i, j):
            return (-1)**(i+j) * abs(self.submatrix(i+1, j+1))

        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        for i in range(self.order[0]):
            for j in range(self.order[1]):
                tmp[j][i] = cofactor(i, j)

        return __class__(tmp)

    @property
    def inverse(self) -> 'Matrix':
        return self.adjoint / abs(self)

    @property
    def null(self) -> 'Matrix':
        return __class__.null_matrix(*self.order)

    @property
    def determinant(self) -> Union[int, float, complex]:
        return abs(self)

    @property
    def as_list(self) -> list[list[Union[int, float, complex]]]:
        return self._matrix

    @property
    def flatten(self) -> tuple[Union[int, float, complex]]:
        return tuple(chain(*self._matrix))

    @property
    def rank(self) -> int:
        ech = self.ElementaryTransform.row_echelon()

        cnt = 0
        for row in ech._matrix:
            if 1 in row:
                cnt += 1

        return cnt

    @classmethod
    def random_matrix(cls, rows: int, cols: int = None, type: str = 'int', num_range: tuple[Union[int, float, complex], Union[int, float, complex]] = None) -> 'Matrix':
        import random

        if cols is None:
            cols = rows
        if num_range is None:
            num_range = (-100, 100) if type == 'int' else (0.0,
                                                           1.0) if type == 'float' else (-10-10j, 10+10j)

        tmp = list([0] * cols for _ in range(rows))

        if type == 'int':
            for i in range(rows):
                for j in range(cols):
                    tmp[i][j] = random.randint(*num_range)

        elif type == 'float':
            for i in range(rows):
                for j in range(cols):
                    tmp[i][j] = random.uniform(*num_range)
        elif type == 'complex':
            for i in range(rows):
                for j in range(cols):
                    tmp[i][j] = complex(random.randint(int(num_range[0].real), int(num_range[1].real)),
                                        random.randint(int(num_range[0].imag), int(num_range[1].imag)))
        else:
            raise MatrixError.UnsupportedType

        return cls(tmp)

    @classmethod
    def row_matrix(cls, row: Iterable[Union[int, float, complex]]) -> 'Matrix':
        return cls([[*row]])

    @classmethod
    def column_matrix(cls, column: Iterable[Union[int, float, complex]]) -> 'Matrix':
        return cls([[i] for i in column])

    @classmethod
    def identity_matrix(cls, size: int) -> 'Matrix':
        tmp = list([0]*size for _ in range(size))
        for i in range(size):
            tmp[i][i] = 1

        return cls(tmp)

    @classmethod
    def null_matrix(cls, rows: int, cols: int = None) -> 'Matrix':
        if cols is None:
            return cls([[0]*rows]*rows)
        else:
            return cls([[0]*cols]*rows)

    def is_same_order(self, other) -> bool:
        return self.order == other.order

    def get_row(self, row: int) -> list[Union[int, float, complex]]:
        # out of range
        return self._matrix[row-1]

    def get_column(self, column: int) -> list[Union[int, float, complex]]:
        return list(row[column-1] for row in self._matrix)

    def get_rows(self, *rows: int) -> tuple[list[Union[int, float, complex]]]:
        return tuple(self.get_row(i) for i in rows)

    def get_columns(self, *columns: int) -> tuple[list[Union[int, float, complex]]]:
        return tuple(self.get_column(i) for i in columns)

    def submatrix(self, rows: int = None, cols: int = None) -> 'Matrix':
        tmp = list([0] * (self.order[1] if cols is None else self.order[1]-1)
                   for _ in range(self.order[0] if rows is None else self.order[0]-1))

        for i in range(len(tmp)):
            for j in range(len(tmp[0])):
                tmp[i][j] = (self._matrix[i if rows is None else i if i < rows-1 else i+1]
                             [j if cols is None else j if j < cols-1 else j+1])

        return __class__(tmp)

    def minors(self, row, col):
        return (-1)**(row+col) * abs(self.submatrix(row, col))

    def cofactor(self, row, col):
        return self.submatrix(row, col)

    def reshape(self, rows: int = None, cols: int = None) -> 'Matrix':
        if rows is None and cols is None:
            rows = self.order[0]
            cols = self.order[1]
        elif rows is None:
            rows = self.order[0]*self.order[1] / cols
            if rows != int(rows):
                raise MatrixError.NotSameElementCount(type=float)
            rows = int(rows)
        elif cols is None:
            cols = self.order[0]*self.order[1] / rows
            if cols != int(cols):
                raise MatrixError.NotSameElementCount(type=float)
            cols = int(cols)

        if self.order[0] * self.order[1] != rows*cols:
            raise MatrixError.NotSameElementCount

        # this is a method to split iterable
        tmp = [iter(self.flatten)]*cols
        return __class__(list(zip(*tmp)))

    @staticmethod
    def is_identical(mat1: 'Matrix', mat2: 'Matrix') -> bool:
        return mat1._matrix == mat2._matrix

    @staticmethod
    def is_same_shape(mat1: 'Matrix', mat2: 'Matrix') -> bool:
        return mat1.order[0] == mat2.order[0] and mat1.order[1] == mat2.order[1]

    @staticmethod
    def Hadamard_product(mat1: 'Matrix', mat2: 'Matrix') -> 'Matrix':
        return mat1 * mat2

    @staticmethod
    def Kronecker_product(mat1: 'Matrix', mat2: 'Matrix') -> 'Matrix':
        tmp = list([0] * (mat1.order[1]*mat2.order[1])
                   for _ in range(mat1.order[0]*mat2.order[0]))

        for index1, elm1 in enumerate(mat1):
            prod1 = elm1 * mat2
            row_start, col_start = (
                (index1)//mat1.order[1]) * mat2.order[0], (index1) % mat1.order[1] * mat2.order[1]

            for i in range(row_start, row_start+mat2.order[0]):
                for j in range(col_start, col_start+mat2.order[1]):
                    tmp[i][j] = prod1._matrix[i-row_start][j-col_start]

        return __class__(tmp)

    def det(mat: 'Matrix') -> Union[int, float, complex]:
        return abs(mat)

    def __str__(self) -> str:   # TODO: implement fallback to str method for terminals that doesn't support unicode
        col_spans = list(len(max(map(str, self.get_column(col+1)), key=len))
                         for col in range(self.order[1]))
        start = f'┏{" "* (self.order[1] + 1 + sum(col_spans))}┓'
        end = f'┗{" "* (self.order[1] + 1 + sum(col_spans))}┛'
        data = [
            f'┃ {" ".join(str(x[i]).rjust(col_spans[i]) for i in range(len(col_spans)))} ┃' for x in self._matrix]

        return '\n'.join([start, *data, end]) + f'{self.order[0]}×{self.order[1]}\n'

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}({self._matrix}) of order {self.order[0]}×{self.order[1]} at {id(self)}>'

    def __abs__(self) -> Union[int, float, complex]:
        if not self.is_square_matrix:
            raise MatrixError.NotSquareMatrix

        if self.order == (2, 2):
            return (self._matrix[0][0] * self._matrix[1][1]) - (self._matrix[1][0] * self._matrix[0][1])

        else:
            return sum((-1)**(2+i)*self._matrix[0][i]*abs(self.submatrix(1, i+1)) for i in range(self.order[1]))

    def __add__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] + other._matrix[i][j]

        elif isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] + other

        return __class__(tmp)

    def __radd__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] + other._matrix[i][j]

        elif isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] + other

        return __class__(tmp)

    def __sub__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] - other._matrix[i][j]

        elif isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] - other

        return __class__(tmp)

    def __rsub__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] - self._matrix[i][j]

        elif isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other - self._matrix[i][j]

        return __class__(tmp)

    def __mul__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other * self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] * other._matrix[i][j]

        return __class__(tmp)

    def __rmul__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other * self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] * other._matrix[i][j]

        return __class__(tmp)

    def __truediv__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] / other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] / other._matrix[i][j]

        return __class__(tmp)

    def __rtruediv__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other / self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] / self._matrix[i][j]

        return __class__(tmp)

    def __floordiv__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] // other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] // other._matrix[i][j]

        return __class__(tmp)

    def __rfloordiv__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other // self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] // self._matrix[i][j]

        return __class__(tmp)

    def __mod__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] % other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] % other._matrix[i][j]

        return __class__(tmp)

    def __rmod__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other % self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] % self._matrix[i][j]

        return __class__(tmp)

    def __divmod__(self, other: Union['Matrix', int, float, complex]) -> tuple['Matrix', 'Matrix']:
        return self.__floordiv__(other), self.__mod__(other)

    def __rdivmod__(self, other: Union['Matrix', int, float, complex]) -> tuple['Matrix', 'Matrix']:
        return self.__rfloordiv__(other), self.__rmod__(other)

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        if not self.order[1] == other.order[0]:
            raise MatrixError.CannotMultiply

        tmp = list([0] * other.order[1] for _ in range(self.order[0]))

        def a(i, j):
            i = self.get_row(i+1)
            j = other.get_column(j+1)

            return sum(map(lambda x, y: x*y, i, j))

        for i in range(self.order[0]):
            for j in range(other.order[1]):
                tmp[i][j] = a(i, j)

        return __class__(tmp)

    def __pow__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] ** other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] ** other._matrix[i][j]

        return __class__(tmp)

    def __rpow__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other ** self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] ** self._matrix[i][j]

        return __class__(tmp)

    def matrix_power(self, power: int) -> 'Matrix':
        if not self.is_square_matrix:
            raise MatrixError.NotSquareMatrix

        if not isinstance(power, int) or 0 > power:
            raise Exception('Power must be a positive integer')

        if power == 0:
            return self.identity

        tmp = self

        for _ in range(power-1):
            tmp @= self

        return tmp

    def __len__(self) -> int:
        return self.order[0] * self.order[1]

    def __eq__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] == other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise Matrix.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] == other._matrix[i][j]

        return __class__(tmp)

    def __lt__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] < other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise Matrix.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] < other._matrix[i][j]

        return __class__(tmp)

    def __le__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] <= other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise Matrix.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] <= other._matrix[i][j]

        return __class__(tmp)

    def __gt__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] > other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise Matrix.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] > other._matrix[i][j]

        return __class__(tmp)

    def __ge__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] >= other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise Matrix.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] >= other._matrix[i][j]

        return __class__(tmp)

    def __ne__(self, other: Union['Matrix', int, float, complex]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int, float, complex)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] != other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise Matrix.NotSameOrder

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] != other._matrix[i][j]

        return __class__(tmp)

    def __bool__(self) -> bool:
        return not 0 in self.flatten

    def __getitem__(self, key: Union[int, slice, tuple[Union[slice, int], Union[slice, int]]]) -> Union['Matrix', int, float, complex]:
        if isinstance(key, int):
            print(key)
            return __class__.row_matrix(self._matrix[key])

        elif isinstance(key, slice):
            return __class__(self._matrix[key])

        elif isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], int):
                if isinstance(key[1], int):
                    return self._matrix[key[0]][key[1]]

                elif isinstance(key[1], slice):
                    return __class__.row_matrix(self._matrix[key[0]][key[1]])

            elif isinstance(key[0], slice):
                if isinstance(key[1], int):
                    return __class__.column_matrix(row[key[1]] for row in self._matrix[key[0]])

                elif isinstance(key[1], slice):
                    return __class__(list(row[key[1]] for row in self._matrix[key[0]]))

        else:
            if not isinstance(key, (int, slice, tuple)):
                raise TypeError(f"indices must be integers or slices.")

            elif isinstance(key, tuple) and len(key) > 2:
                raise IndexError

    def __setitem__(self, key: Union[int, slice, tuple[Union[int, slice], Union[int, slice]]], value: Union[int, float, complex, 'Matrix', Iterable]) -> None:
        if not isinstance(value, (int, float, complex, Matrix, Iterable)):
            raise ValueError

        if isinstance(key, int) and isinstance(value, (Matrix, Iterable)):
            if len(value) == self.order[1]:
                self._matrix[key] = list(value)
            else:
                raise ValueError(
                    "Number of elements should be equal to number of columns of the matrix")

        elif isinstance(key, slice) and isinstance(value, Iterable) and isinstance(value[0], Iterable):
            if len(self._matrix[key]) == len(value) and all(len(i) == self.order[1] for i in value):
                self._matrix[key] = value

            else:
                raise ValueError(
                    "Number of elements should be equal to number of columns of the matrix and equal number of rows of the matrix row slice")

        # exceptions not handled
        elif isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], int):
                if isinstance(key[1], int):
                    self._matrix[key[0]][key[1]] = value

                elif isinstance(key[1], slice):
                    self._matrix[key[0]][key[1]] = value

            elif isinstance(key[0], slice):
                if isinstance(key[1], int):
                    if isinstance(value[0], Iterable):
                        for i in range(len(value)):
                            self._matrix[key[0]][i][key[1]] = value[i][0]

                    else:
                        for i in range(len(value)):
                            self._matrix[key[0]][i][key[1]] = value[i]

                elif isinstance(key[1], slice):
                    for i in range(len(value)):
                        self._matrix[key[0]][i][key[1]] = value[i]

    def __iter__(self) -> Iterable[Union[int, float, complex]]:
        return iter(self._matrix)

    def __reversed__(self) -> Iterable[Union[int, float, complex]]:
        return self.flatten[::-1]

    def __contains__(self, key: Union[int, float, complex]) -> bool:
        return key in self.flatten

    def __pos__(self) -> 'Matrix':
        return self

    def __neg__(self) -> 'Matrix':
        return -1 * self

    def __invert__(self) -> 'Matrix':
        if not all(isinstance(i, int) for i in self):
            raise MatrixError.ElementTypeError

        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        for i in range(self.order[0]):
            for j in range(self.order[1]):
                tmp[i][j] = ~self._matrix[i][j]

        return __class__(tmp)

    def __or__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] | other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] | other._matrix[i][j]

        return __class__(tmp)

    def __ror__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other | self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] | self._matrix[i][j]

        return __class__(tmp)

    def __and__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] & other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] & other._matrix[i][j]

        return __class__(tmp)

    def __rand__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other & self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] & self._matrix[i][j]

        return __class__(tmp)

    def __xor__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] ^ other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] * other._matrix[i][j]

        return __class__(tmp)

    def __rxor__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other ^ self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] ^ self._matrix[i][j]

        return __class__(tmp)

    def __rshift__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] >> other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] >> other._matrix[i][j]

        return __class__(tmp)

    def __rrshift__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other >> self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] >> self._matrix[i][j]

        return __class__(tmp)

    def __lshift__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] << other

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = self._matrix[i][j] << other._matrix[i][j]

        return __class__(tmp)

    def __rlshift__(self, other: Union['Matrix', int]) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        if isinstance(other, (int)):
            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other << self._matrix[i][j]

        elif isinstance(other, Matrix):
            if not self.order == other.order:
                raise MatrixError.NotSameOrder

            if not (any(isinstance(i, int) for i in self) and any(isinstance(i, int) for i in other)):
                raise MatrixError.ElementTypeError

            for i in range(self.order[0]):
                for j in range(self.order[1]):
                    tmp[i][j] = other._matrix[i][j] << self._matrix[i][j]

        return __class__(tmp)

    def __round__(self, precision: int = None) -> 'Matrix':
        tmp = list([0] * self.order[1] for _ in range(self.order[0]))

        for i in range(self.order[0]):
            for j in range(self.order[1]):
                tmp[i][j] = round(self._matrix[i][j], precision)

        return __class__(tmp)

    class __ElementaryTransform:
        def __init__(self, matrix: 'Matrix'):
            self.matrix = matrix

        # TODO: implement elementary transform
        def row_echelon(self) -> 'Matrix':
            tmp = deepcopy(self.matrix._matrix)
            tmp = list(list(map(lambda x: Decimal(str(x)), row))
                       for row in tmp)

            def sort():
                nonlocal tmp
                def z_s(a):
                    cnt = 0
                    for i in a:
                        if i == 0:
                            cnt += 1
                        else:
                            break
                    return cnt
                ech_ord = [(index, z_s(row)) for index, row in enumerate(tmp)]
                ech_ord.sort(key=lambda x: x[1])
                tmp = [tmp[i[0]] for i in ech_ord]
            
            def divide(_from = None):
                nonlocal tmp
                for row in tmp[_from:]:
                    for i in row:
                        if i:
                            num = i
                            break
                    else: continue
                    if num == 1: continue
                    for index, i in enumerate(row):
                        row[index] = i/num if i else i
            
            def row_reduce(_from = 0):
                nonlocal tmp
                if _from == len(tmp):
                    return
                ref = tmp[_from]
                for index, row in enumerate(tmp[_from+1:]):
                    if row[_from] == 0: continue
                    tmp[index + _from + 1] = [(i - ref[loc]) for loc, i in enumerate(row)]
                    
            for i in range(len(tmp)):
                sort()
                divide(i)
                row_reduce(i)
                
            tmp = list((list(map(lambda x: int(z) if ('.' not in (
                z := x.to_eng_string())) else float(z), row)) for row in tmp))
            return Matrix(tmp)

        def column_echelon(self) -> 'Matrix':
            pass

        def row_multiply(self, row: int, mul: Union[int, float]) -> 'Matrix':
            mat = self.matrix._matrix.copy()
            mat[row-1] = [i*mul for i in mat[row-1]]
            return __class__(mat)

        def column_multiply(self, col: int, mul: Union[int, float]) -> 'Matrix':
            mat = deepcopy(self.matrix._matrix)
            for row in mat:
                row[col-1] = row[col-1]*mul

            return __class__(mat)

        def row_transform(self, R_n1: int, x: Union[int, float], R_n2: int) -> 'Matrix':
            mat = deepcopy(self.matrix._matrix)
            mat[R_n1-1] = [i+x*j for i, j in zip(mat[R_n1-1], mat[R_n2-1])]

            return __class__(mat)

        def column_transform(self, C_n1: int, x: Union[int, float], C_n2: int) -> 'Matrix':
            mat = deepcopy(self.matrix._matrix)
            for row in mat:
                row[C_n1-1] = row[C_n1-1] + x*row[C_n2-1]

            return __class__(mat)

        def row_swap(self, R_n1: int, R_n2: int) -> 'Matrix':
            mat = deepcopy(self.matrix._matrix)
            mat[R_n1-1], mat[R_n2-1] = mat[R_n2-1], mat[R_n1-1]

            return __class__(mat)

        def column_swap(self, C_n1: int, C_n2: int) -> 'Matrix':
            mat = deepcopy(self.matrix._matrix)
            for row in mat:
                row[C_n1-1], row[C_n2-1] = row[C_n2-1], row[C_n1-1]

            return __class__(mat)

        # def _row_multiply(row, mul, mat):
        #     mat[row-1] = [i*mul for i in mat[row-1]]

        # def _column_multiply(col, mul, mat):
        #     for row in mat:
        #         row[col-1] = mul*row[col-1]

        # def _row_transform(R_n1, x, R_n2, mat):
        #     mat[R_n1-1] = [i+x*j for i, j in zip(mat[R_n1-1], mat[R_n2-1])]

        # def _column_transform(C_n1, x, C_n2, mat):
        #     for row in mat:
        #         row[C_n1-1] = row[C_n1-1] + x*row[C_n2-1]

        # def _row_swap(R_n1, R_n2, mat):
        #     mat[R_n1-1], mat[R_n2-1] = mat[R_n2-1], mat[R_n1-1]

        # def _column_swap(C_n1, C_n2, mat):
        #     for row in mat:
        #         row[C_n1-1], row[C_n2-1] = row[C_n2-1], row[C_n1-1]

    class __LinearTransform:    # TODO: linear transform not done
        def __init__(self, matrix: 'Matrix'):
            pass

        def shear(self, axis: Union[int, str], value: Union[int, float]) -> 'Matrix':
            pass

        def reflection(self, axis: Union[int, str], value: Union[int, float]) -> 'Matrix':
            pass

        def squeeze_map(self, ratio: Union[int, float]) -> 'Matrix':
            pass

        def scale(self, scale: Union[int, float]) -> 'Matrix':
            pass

        def rotate(self, angle: Union[int, float]):
            pass
