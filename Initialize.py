import numpy as np
import time

def Data_Shape(Data):
    List_A = []
    List_B = []
    for row in range(Data.shape[0]):
        List_A.append(Data[row][0])
        List_B.append(Data[row][1])
    List_A = list(set(List_A))
    List_B = list(set(List_B))
    length_A = len(List_A)
    length_B = len(List_B)
    print('Data set length:'+str(Data.shape[0]))
    print('Node length of the first column:('+str(length_A)+')')
    print('Node length of the second column:('+str(length_B)+')')
    MaxNodeNum =  int(max(max(List_A),max(List_B)))+1
    print('Number of nodes is:'+str(MaxNodeNum))
    return MaxNodeNum
def MatrixAdjacency(MaxNodeNum,Data):
    MatrixAdjacency = np.zeros([MaxNodeNum,MaxNodeNum])
    for col in range(Data.shape[0]):
        i = int(Data[col][0])
        j = int(Data[col][1])
        MatrixAdjacency[i,j] = 1
        MatrixAdjacency[j,i] = 1
    return MatrixAdjacency
def spones(Array):
    for index in range(len(Array)):
        if Array[index] != 0:
            Array[index] = 1
    return Array
def writeTrainFile(Matrix,NetName):
    Matrix = np.triu(Matrix)
    index = np.argwhere(Matrix != 0)
    with open('Data\\'+NetName+'\\Train.txt','w') as file:
        np.savetxt(file,index,fmt='%d')
def writeTestFile(Matrix,NetName):
    Matrix = np.triu(Matrix)
    index = np.argwhere(Matrix != 0)
    with open('Data\\'+NetName+'\\Test.txt','w') as file:
        np.savetxt(file,index,fmt='%d')
def Init(NetFile):
    print("DataShape......")
    NetData = np.loadtxt(NetFile)
    MaxNodeNum = Data_Shape(NetData)
    MatrixAdjacency_Net = MatrixAdjacency(MaxNodeNum, NetData)
    return MatrixAdjacency_Net,MaxNodeNum

def Divide(NetFile,MatrixAdjacency_Net,MaxNodeNum,NetName):
    print("Divide......")
    DivideTime_Start = time.time_ns()
    DivideNum = 0.9
    NetData = np.loadtxt(NetFile)
    lengthData = len(NetData)
    NumSet = NetData.shape[0]
    NumTest = int(float(1-DivideNum)*NumSet)
    MatrixAdjacency_Test =np.zeros([MaxNodeNum,MaxNodeNum])

    while(len(np.nonzero(MatrixAdjacency_Test)[0]) < NumTest):
        index_Link = int(np.random.rand(1) * NetData.shape[0])
        Uid1 = int(NetData[index_Link,0])
        Uid2 = int(NetData[index_Link,1])

        MatrixAdjacency_Net[Uid1,Uid2] = 0
        MatrixAdjacency_Net[Uid2,Uid1] = 0

        tempVector = MatrixAdjacency_Net[Uid1]
        sign = 0
        Uid1_TO_Uid2 = np.dot(tempVector,MatrixAdjacency_Net) + tempVector
        if Uid1_TO_Uid2[Uid2] > 0:
            sign = 1
        else:
            count = 1
              #for x in range(MatrixAdjacency_Net.shape[0]):
                  #print (len((spones(Uid1_TO_Uid2) - tempVector).nonzero()[0]))
            while (len((spones(Uid1_TO_Uid2) - tempVector).nonzero()[0]) != 0):
              #while((spones(Uid1_TO_Uid2) - tempVector).nonzero() != 0):
                  #print( "n steps up to")
                  #print( [Uid1,Uid2])
                  #print( Uid1_TO_Uid2[Uid2])
                  #print( tempVector.nonzero())
                  #print( spones(Uid1_TO_Uid2).nonzero())
                  #print( np.nonzero(spones(Uid1_TO_Uid2) - tempVector))
                  #print( (spones(Uid1_TO_Uid2) - tempVector).nonzero())
                tempVector  = spones(Uid1_TO_Uid2)
                Uid1_TO_Uid2 = np.dot(tempVector,MatrixAdjacency_Net) + tempVector
                count += 1
                if Uid1_TO_Uid2[Uid2] > 0:
                    sign = 1
                    break
                if count >= MatrixAdjacency_Net.shape[0]:
                    print('Unreachable:'+str([Uid1,Uid2]))
                    sign = 0
        if sign == 1:
            NetData = np.delete(NetData,index_Link,axis=0)
            MatrixAdjacency_Test[Uid1,Uid2] = 1
        else:
            NetData = np.delete(NetData,index_Link,axis=0)
            MatrixAdjacency_Net[Uid1,Uid2] = 1
            MatrixAdjacency_Net[Uid2,Uid1] = 1

    MatrixAdjacency_Train = MatrixAdjacency_Net
    MatrixAdjacency_Test = MatrixAdjacency_Test + MatrixAdjacency_Test.T

    print('Training set adjacency matrix:'+str(MatrixAdjacency_Train.shape))
    print('Training set plan number of edges:'+str(lengthData - NumTest))
    print('Actual number of edges in training set:'+str((np.nonzero(MatrixAdjacency_Train)[0].shape[0])/2))
    print('Test set adjacency matrix:'+str(MatrixAdjacency_Test.shape))
    print('Test set plan number of edges:'+str(NumTest))
    print('Actual number of edges in the test set:'+str((np.nonzero(MatrixAdjacency_Test)[0].shape[0])/2))


    writeTrainFile(MatrixAdjacency_Train, NetName)
    writeTestFile(MatrixAdjacency_Test, NetName)
    DivideTime_End = time.time_ns()
    print('DivideTime：%f s'%(DivideTime_End - DivideTime_Start))
    return MatrixAdjacency_Train,MatrixAdjacency_Test

def Init2(Test_File,Train_File):
    print("DataShape......")
    TrainData = np.loadtxt(Train_File)
    MaxNodeNumTrain = Data_Shape(TrainData)
    TestData = np.loadtxt(Test_File)
    MaxNodeNumTest = Data_Shape(TestData)
    MaxNodeNum = max(MaxNodeNumTrain,MaxNodeNumTest)

    MatrixAdjacency_Train = MatrixAdjacency(MaxNodeNum, TrainData)
    MatrixAdjacency_Test = MatrixAdjacency(MaxNodeNum, TestData)

    return MatrixAdjacency_Train,MatrixAdjacency_Test,MaxNodeNum
