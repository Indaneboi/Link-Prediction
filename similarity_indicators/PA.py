#coding=UTF-8
import numpy as np
import time

def PA(MatrixAdjacency_Train):
    similarity_StartTime = time.time_ns()

    deg_row = sum(MatrixAdjacency_Train)
    deg_row.shape = (deg_row.shape[0],1)
    deg_row_T = deg_row.T

    Matrix_similarity = np.dot(deg_row,deg_row_T)

    similarity_EndTime = time.time_ns()
    print("Preferential Attachment Similarity Time: %f s" % (similarity_EndTime- similarity_StartTime))
    return Matrix_similarity
