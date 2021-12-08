# Matrix

> A python module containing most of the matrix properties and operations<br/>
> ! since this is implemented in vanilla python using it for large scale project will be very inefficient

---

## Installation

Just download the matrix.py file to the project directory 😁

---

## Usage

- ### importing

    ```python
    from matrix import Matrix
    ```

- ### creating a new matrix
  
  ```python
  matrix_data: list[list[int | float | complex]] = [    # every row should've equal number of elements
                                                        [...],
                                                        [...],
                                                        [...], ...
                                                   ]
  Matrix(matrix_data)

  # Alt
  Matrix.random_matrix(3, 3, 'int')     # matrix with random values (3×3 integer)
  Matrix.row_matrix([...])              # row matrix
  Matrix.column_matrix([...])           # column matrix
  Matrix.identity_matrix(3))            # identity matrix (3×3)
  Matrix.null_matrix(3, 3)              # null matrix (3×3)
  ```

- ### Operations

  ```python
  
  ```
<!-- TODO: fill this shit -->