# numc

### Overview
-  Developed a high-performance version of NumPy in C, leveraging the Python-C interface and optimizing with SIMD, OpenMP, and cache blocking—resulting in a staggering 1300x speed increase in matrix operations.
- Engineered core functionalities including addition, multiplication, absolute value computation, power operations, and advanced matrix slicing for both rows and columns.
- Ensured software robustness by authoring comprehensive unit tests, achieving a branch coverage of 100%.

### Documentation

#### Summary
`numc` is a libarary using for basic matrix calculation operations. Numc is implemented by Python-C interface. Numc provides extremely efficient operations for matrix. Numc supports matrix addition, matrix multiplication, absolute matrix calculation and power calculation. Numc also supports matrix slicing, getting a value from ith row jth col of a matrix, and setting a value at ith row jth col of a matrix. 

To use `numc`:
```python
>>> import numc as nc
```
And you will see,
```
CS61C Summer 2021 Project 4: numc imported!
```
`numc` is imported successfully, feel free to use `numc` doing your matrix operations!


#### File Structures
`src`:
- `matrix.c`: functions used to implement matrix operations.
- `matrix.h`: header file for `matrix.c`, including struct and function definitions.
- `numc.c`: Python-C interface, dealing with exceptions.
- `numc.h`: header file for `num.c`.

`tests`
- `/unitests`: tests for what we write for numc, including for small, medium and large matrix, used to compare.
- `mat_test.c`: tests for basic matrix operation.
