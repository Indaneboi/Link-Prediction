#coding=UTF-8
import numpy as np
import time

def Cn(MatrixAdjacency_Train):
    similarity_StartTime = time.time_ns()

    Matrix_similarity = np.dot(MatrixAdjacency_Train,MatrixAdjacency_Train)

    similarity_EndTime = time.time_ns()
    print("Common neighbor Similarity Time: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity
