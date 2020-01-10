import pickle
from copy import deepcopy
from dataGeneration import dataset


def loadReconstruction(fileName):
    return pickle.load(open('reconstructions/'+str(fileName)+'_reconstr.pickle','rb'))

class reconstruction():

    def __init__(self,functionIn, ID = 'untitled_reconstruction',*reconstrArgs):
        self.ID = ID
        self.reconstructionFunction = deepcopy(functionIn)
        self.otherArgs = reconstrArgs
        if reconstrArgs == (): #taking care of the no-additional-arguments case:
            self.otherArgs = [None]

    def info(self):
        print('Reconstruction ID: ' + str(self.ID))

    def reconstruct(self,yourDatasetObj):
        return self.reconstructionFunction(yourDatasetObj, *self.otherArgs)

    def save(self,fileName = None):
        if fileName == None:
            pickle.dump(self,open('reconstructions/'+str(self.ID)+'_reconstr.pickle','wb'))
        else:
            pickle.dump(self,open('reconstructions/'+str(fileName)+'_reconstr.pickle','wb'))




# Recommended Reconstruction Functions...-----------------------------------------------------------------

'''
def reconstruction_template(datasetObj,arg1,arg2,...):
    recDataName = ''
    recData = f(datasetObj.data)
    recCompNames = g(datasetObj.componentNames)
    datasetToReturn = dataset(recData,recDataName,recCompNames)
    datasetToReturn.dt = datasetObj.dt
    return datasetToReturn

'''

def add1(datasetObj,otherArgs):
    data = datasetObj.data
    recData = [[data[timepoint][component]+1 for component in range(len(data[0]))] for timepoint in range(len(data))]
    recCompNames = [name + '_add1' for name in datasetObj.componentNames]
    return dataset('add1Example',recData,recCompNames)

def firstNLagsGivenTau(datasetObj,nLags,tau):
    '''Takes data and returns original and timelags specified by nLags, tau.

    If my data is [[1,2,3,4,5,6,7]], and I choose tau=2, nlags = 2, this returns for me:
    [[5,6,7],[3,4,5],[1,2,3]]. The first component is the "original;' the second is the "first lag at length 1*tau";
    the final is the "second lag at length 2*tau"

    :param compData: A list of lists of time series data, indexed first by component and second by time.
    :param nLags: The number of lags for a given component. The dimension of reconstrtucted data is (nLags+1)*len(compData)
    :param tau: The "length" of the window to the past.
    :return: trimmedData, a list of lists of even length of component data and lags. lags of a given length are grouped together.
    '''
    timeData = datasetObj.data
    compData = switchDataIndices(timeData)
    numComps = len(compData)
    allData = compData.copy()
    #Acquiring the lags
    for k in range(int(nLags)):
        lagsHere = []
        for i in range(numComps):
            lagsHere.append(allData[-numComps+i][0:-tau])
        for elem in lagsHere: allData.append(elem)
    #Trimming the ends off the lags
    npoints = len(allData[-1])
    trimmedData = []
    for row in allData:
        trimmedData.append(row[-npoints:])
    trimmedData = switchDataIndices(trimmedData)

    recDataSetName = str(datasetObj.ID) + '_lags_N'+str(nLags) + '_tau' + str(tau)
    recComponentNames = datasetObj.componentNames
    for i in range(nLags):
        recComponentNames = recComponentNames + ['lag'+str(i+1)+datasetObj.componentNames[j] for j in range(len(datasetObj.componentNames))]

    datasetToReturn = dataset(trimmedData,recDataSetName,recComponentNames)
    datasetToReturn.dt = datasetObj.dt
    return datasetToReturn



def reconstructOriginalAndNDerivatives(datasetObj,nDeriv,dt=1,typeHere='left'):
    '''Repeatedly applies reconstructOriginalAndFirstDerivative to get multiple derivatives of time series data.

    :param compdata: A list of lists of time series data, indexed first by component and second by time.
    :param dt: The sampling time for this dataset. If in doubt, set to .01 or something.
    :param nDeriv: The number of derivatives to compute for each component. Reconstructed dim is len(compData)*(nDeriv+1)
    :param typeHere: A toggle for the derivative type; pick either "center" or "left.'
    :return:
    '''
    timeData = datasetObj.data
    compdata = switchDataIndices(timeData)
    numComps = len(compdata)
    dataOut = compdata
    #Getting the derivatives
    for i in range(nDeriv):
        newDeriv = reconstructOriginalAndFirstDerivative(dataOut[-numComps:],dt,typeHere)
        dataOut = dataOut + newDeriv[-numComps:]
    #trimming the ends
    lastrowlen = len(dataOut[-1])
    cutoff = [elem[-lastrowlen:] for elem in dataOut]
    cutoff = switchDataIndices(cutoff)

    recDataSetName = str(datasetObj.ID) + '_Derivs_N'+str(nDeriv) + '_dt' + str(dt)
    recComponentNames = datasetObj.componentNames
    for i in range(nDeriv):
        recComponentNames = recComponentNames + [str(i+1)+'dot'+datasetObj.componentNames[j] for j in range(len(datasetObj.componentNames))]

    datasetToReturn = dataset(cutoff,recDataSetName,recComponentNames)
    datasetToReturn.dt = datasetObj.dt
    return datasetToReturn


#Functions used in the recommended reconstruction functions above...
def reconstructOriginalAndFirstDerivative(compdata,dt,typeHere = 'left'): #'center','left'
    '''Gets the center-or-left first derivatives of a time-series dataset, returning both the original data and the derivative.

    :param compdata: A list of lists of time series data, indexed first by component and second by time.
    :param dt: The sampling time for this dataset. If in doubt, set to .01 or something.
    :param typeHere: A toggle for the derivative type; pick either "center" or "left.'
    :return:
    '''

    numComps = len(compdata)
    allDerivatives = []
    cutOffCompData = []
    if typeHere == 'left':
        for i in range(numComps):
            row = []
            for j in range(len(compdata[i])-1):
                row.append((compdata[i][j+1]-compdata[i][j])/dt)
            allDerivatives.append(row)
        for i in range(numComps):
            cutOffCompData.append(compdata[i][1:])
    elif typeHere == 'symmetric':
        for i in range(numComps):
            row = []
            for j in range(1,len(compdata[i])-1):
                row.append((compdata[i][j+1]-compdata[i][j-1])/(2*dt))
            allDerivatives.append(row)
        for i in range(numComps):
            cutOffCompData.append(compdata[i][1:-1])
    else:
        print('type not recongnized.')
        print(typeHere)
    return cutOffCompData + allDerivatives

def switchDataIndices(compData):
    dataByTime = []
    for i in range(len(compData[0])):
        rowhere = []
        for j in range(len(compData)):
            rowhere.append(compData[j][i])
        dataByTime.append(rowhere)
    return dataByTime

'''
myDataset = dataset([[1,10],[3,30],[5,50],[7,70],[9,90],[11,110],[13,130],[15,150],[17,170]])
#myReconstr = reconstruction(firstNLagsGivenTau,'0-1Lags,Tau=1',1,1)
myReconstr = reconstruction(reconstructOriginalAndNDerivatives,'0-2Derivatives,dt=1',2,.1,'left')
#myReconstr = reconstruction(add1,'Add_one_to_all')
outputBoi = myReconstr.reconstruct(myDataset)
outputBoi.info()
myReconstr.save()
outputBoi.plot([0,1,2])
'''


'''
newReconstr = loadReconstr('0-1Derivatives,dt=1')
print('down here')
print(newReconstr.reconstruct(myData))
'''