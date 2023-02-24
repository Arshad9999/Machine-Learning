
import numpy as np
matrix1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
matrix2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
print("Matrix1 : \n", matrix1)
print("Matrix2 : \n", matrix2)
print("Transpose of a matrix1 : \n", matrix1.T)
print("Transpose of a matrix2 : \n", matrix2.T)
print("Addition of matrix1 and matrix2 : \n", matrix1 + matrix2)
print("Substraction of matrix2 from matrix1 : \n", matrix1 - matrix2)
print("Mutliplication of matrix1 and matrix2 : \n", matrix1 @ matrix2)

