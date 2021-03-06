#coding=UTF-8
import numpy as np
import time

def Katz(MatrixAdjacency_Train):
    similarity_StartTime = time.time_ns()

    Parameter = 0.01
    Matrix_EYE = np.eye(MatrixAdjacency_Train.shape[0])
    Temp = Matrix_EYE - MatrixAdjacency_Train * Parameter
    Matrix_similarity = np.linalg.inv(Temp)
    Matrix_similarity = Matrix_similarity - Matrix_EYE

    similarity_EndTime = time.time_ns()
    print("Katz Similarity Time: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity
