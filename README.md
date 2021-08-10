# numc

### Provide answers to the following questions for each set of partners.
- How many hours did you spend on the following tasks?
  - Task 1 (Matrix functions in C): We spend 20 hours on Task 1, which is Matrix functions in C.
  - Task 2 (Writing the Python-C interface): We also spend 20 hours on Task2, which is Writing the Python-C interface.
  - Task 3 (Speeding up matrix operations): We spend 30 hours on Task 3, which is Speeding up matrix operations.
- Was this project interesting? What was the most interesting aspect about it?
  - <b>This is the most interesting project we have ever done! We both like it so much. I am so happy that I truly learned a lot from this project.</b>
- What did you learn?
  - <b>I learned about how C is connected with Python. The Pyobeject is so powerful.</b>
  - <b>We also tried different ways to speed up our operations, including reducing function calls and using different algorithms , utilizing SIMD and OpenMP.</b>
- Is there anything you would change?
  - <b>I think our multiply function is not good enough. I think there must be an algorithm for matrix multiply that can achieve higher speed up.</b>
  - <b>I think our power function is not good enough. We found a super efficient algorithm on Ed but we think it is too hard for us to implement.</b>

### If you worked with a partner:
- In one short paragraph, describe your contribution(s) to the project.
  - <b>We use Visual Studio Code. We use the Visual Studio Live Share to work together on the project. We also working in our private zoom meeting so that we can share our thoughts when we are working on the project. For matrix.c file, Jiaxin mainly works on writing the code. And Zuhang mainly works on maintaining if the code is correct and the logic is correct. Jiaxin come up with an great algorithm for matrix multiplication. And Zuhang come up with an great algorithm for power multiplication. We both work hard to delete many unneccessary function call. Zuhang found that original naive power function is not fast enough. Then he come up with a faster algorithm for matrix pow function. </b>
- In one short paragraph, describe your partner's contribution(s) to the project.
  - <b>We worked together when we were coding the project. We disscussed a lot on what we can to do speed up the operations. We debugged together, and did our best to try different ways to meet the benchmark. I feel so happy to work with my partner. For numc.c file part, Jiaxin works on unpacking the arguments and connecting C with python. Zuhang works on throwing all errors. For the most part the project, we work together using Visual Studio Live Share. </b>

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
- `numc.c`: Python-C interface, dealing withexceptions.
- `numc.h`: header file for `num.c`.

`tests`
- `/unitests`: tests for what we write for numc, including for small, medium and large matrix, used to compare.
- `mat_test.c`: tests for basic matrix operation.