import time
import os
import Initialize
import Evaluation_Indicators.AUC


import similarity_indicators.CommonNeighbor
import similarity_indicators.Jaccard
import similarity_indicators.PA
import similarity_indicators.AA
import similarity_indicators.HDI
import similarity_indicators.HPI

import similarity_indicators.Katz
import similarity_indicators.ACT

startTime = time.time_ns()
#Initialize the training test set

README1 = '''Please choose a DataSet:
    WIKI           1
    FACEBOOK       2
    EMAIL-EU       3'''

print(README1)
Set = int(input('Input Set:'))
if Set == 1:
    NetFile = u'Data/WIKI.txt'
    NetName = 'WIKI'
elif Set == 2:
    NetFile = u'Data/FACEBOOK.txt'
    NetName = 'FACEBOOK'
elif Set == 3:
    NetFile = u'Data/EMAIL-EU.txt'
    NetName = 'EMAIL-EU'
else:
    print('Input Error')

print ("\nLink Prediction start：\n")
TrainFile_Path = 'Data\\'+NetName+'\\Train.txt'
if os.path.exists(TrainFile_Path):
    Train_File = 'Data\\'+NetName+'\\Train.txt'
    Test_File = 'Data\\'+NetName+'\\Test.txt'
    MatrixAdjacency_Train,MatrixAdjacency_Test,MaxNodeNum = Initialize.Init2(Test_File, Train_File)
else:
    MatrixAdjacency_Net,MaxNodeNum = Initialize.Init(NetFile)
    MatrixAdjacency_Train,MatrixAdjacency_Test = Initialize.Divide(NetFile, MatrixAdjacency_Net, MaxNodeNum,NetName)

#Similarity matrix calculation

# README = '''\nPlease choose a method:
#     CN            0
#     Jaccard       1
#     PA            2
#     AA            3
#     Katz          4
#     ACT           5'''
# print (README)
# Method = int(input('Input Method:'))

# Matrix_similarity = similarity_indicators.Cos.ACT(MatrixAdjacency_Train)

print('--------------------Node based similarity--------------------')
print('----------Common Neighborhood----------')
Matrix_similarity = similarity_indicators.CommonNeighbor.Cn(MatrixAdjacency_Train)
Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)
print('----------Jaccard----------')
Matrix_similarity = similarity_indicators.Jaccard.Jaccards(MatrixAdjacency_Train)
Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)
print('----------Prefential Attachment----------')
Matrix_similarity = similarity_indicators.PA.PA(MatrixAdjacency_Train)
Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)
#print('----------Adamic Adar----------')
#Matrix_similarity = similarity_indicators.AA.AA(MatrixAdjacency_Train)
#Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)
print('----------HDI----------')
Matrix_similarity = similarity_indicators.HDI.HDI(MatrixAdjacency_Train)
Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)
print('----------HPI----------')
Matrix_similarity = similarity_indicators.HPI.HPI(MatrixAdjacency_Train)
Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)
print('--------------------Path based similarity--------------------')
print('----------Katz----------')
Matrix_similarity = similarity_indicators.Katz.Katz(MatrixAdjacency_Train)
Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)
print('----------Commute Time----------')
Matrix_similarity = similarity_indicators.ACT.ACT(MatrixAdjacency_Train)
Evaluation_Indicators.AUC.Calculation_AUC(MatrixAdjacency_Train, MatrixAdjacency_Test, Matrix_similarity, MaxNodeNum)



endTime = time.time_ns()
print("\nRunTime: %f s" % (endTime - startTime))
