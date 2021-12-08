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

  - supported operators :-

    >  **+&nbsp;&nbsp;&nbsp;-&nbsp;&nbsp;&nbsp;*&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;//&nbsp;&nbsp;&nbsp;%&nbsp;&nbsp;&nbsp;\*\*&nbsp;&nbsp;&nbsp;^&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&&nbsp;&nbsp;&nbsp;>>&nbsp;&nbsp;&nbsp;<<&nbsp;&nbsp;&nbsp;~&nbsp;&nbsp;&nbsp;==&nbsp;&nbsp;&nbsp;>&nbsp;&nbsp;&nbsp;<&nbsp;&nbsp;&nbsp;<=&nbsp;&nbsp;&nbsp;>=&nbsp;&nbsp;&nbsp;!=**
    >
    > - if left and right are both matrices (both should be in same order) -> do the operation to corresponding elements.
    > - if one side is not a matrix(int, float, complex,...) -> apply the operator to every element of the matrix with that one value.

  - **@** (matrix multiplication)
  - **matrix_power** (matrix power from matrix multiplication)
  - **Kronecker_product**
  - **Hadamard_product**
  - **reshape**
  - **cofactor**
  - **minors**
  - **submatrix**
  - **round**

- ### Properties
  
  - **order**
  - **bool**

- ### Elementary Transform

  - **row_echelon**
  - **column_echelon**
  - **row_multiply**
  - **column_multiply**
  - **row_transform**
  - **column_transform**
  - **row_swap**
  - **column_swap**
