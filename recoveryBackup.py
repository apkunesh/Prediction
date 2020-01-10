import numpy as np
import pickle
import sympy
from sympy import lambdify
from sympy import sin as symsin
from sympy import cos as symcos
import itertools, operator
from dataGeneration import *
from time import time
from numpy.linalg import lstsq as leastSquares

def loadModel(modelName):
    return pickle.load(open('models/'+str(modelName)+'_model.pickle','rb'))


def getBasis(basisType, orderInside, modelSymbols):
    '''Gets a sympy basis to be used for construction of a predictive model/fitting to some time series data.

    :param basisType: A string indicating a mathematical basis from which to recover the Equations of Motion.
    :param orderInside: An int indicating the maximum order for the basis elements.
    :param modelSymbols: The sympy variables for use in the model.
    :return: basisOut, a list of sympy basis functions up to order *orderInside
    '''
    orderInside = int(orderInside)
    if basisType == 'Taylor':
        basisOut = taylorBasisGivenOrder(orderInside, modelSymbols)
    elif basisType == 'Fourier':
        basisOut = fourierBasisGivenOrder(orderInside, modelSymbols)
    return basisOut

def numNonzeroElems(twoDList):
    n = 0
    for i in range(len(twoDList)):
        for j in range(len(twoDList)):
            if twoDList[i][j]!=0:
                n+=1
    return n

def round_sig(x, sig=5): #will fail if x=0
    if round(x) != 0:
        return round(x, sig-int(np.floor(np.log10(abs(x))))-1)
    else:
        return 0

def combinations_with_replacement_counts(n, r):
    '''A function to enumerate the possible configurations of r indistinguishable balls into n distinguishable bins.

    Used to produce all basis functions at some order r for a number of dimensions n.

    :param n: int number of dimensions of your basis, or, generally, number of distinguishable bins
    :param r: int order of a basis function, or, generally, number of indistinguishable balls
    :return: a tuple of realizations of balls-in-bins.
    '''
    size = n + r - 1
    for indices in itertools.combinations(range(size), n-1):
        starts = [0] + [index+1 for index in indices]
        stops = indices + (size,)
        yield tuple(map(operator.sub, stops, starts))

def fourierBasisGivenOrder(orderInside,modelSymbols):
    '''Gets a list of sympy trigonometric basis functions for use in curve fitting.

    :param orderInside: An int indicating the maximum order for the basis elements.
    :param modelSymbols: The sympy variables for use in the model.
    :return: basis, a list of sympy basis functions (here, sine and cosine) up to order *orderInside
    '''
    dim = len(modelSymbols)
    modelSymbols.reverse()
    basis = [1]
    allCoeffs = []
    for k in range(orderInside+1):
        multiplesAtOrderKTuple = list(combinations_with_replacement_counts(dim,k))
        listForm = [i[j] for i in multiplesAtOrderKTuple for j in range(len(i))]
        allCoeffs = allCoeffs + listForm
    separated = []
    for j in range(int(len(allCoeffs)/dim)):
        separated.append(allCoeffs[j*dim:dim*(j+1)])
    for count in range(len(separated)):
        insideTrig = sum([nlm * symbol for nlm,symbol in zip(separated[count],modelSymbols)])
        basis.append(symcos(insideTrig))
        basis.append(symsin(insideTrig))
    basis.remove(0)
    basis.pop(1)
    return basis

def taylorBasisGivenOrder(taylorOrder,mbedSymbols):
    '''Gets a list of sympy polynomial basis functions for use in curve fitting.

    :param orderInside: An int indicating the maximum order for the basis elements.
    :param modelSymbols: The sympy variables for use in the model.
    :return: basis, a list of sympy basis functions (here, 1, x, x^2, y, y^2, xy...) up to order *orderInside
    '''
    taylorOrder = taylorOrder+1
    dim = len(mbedSymbols)
    basis = []
    allExponents = []
    for k in range(taylorOrder):
        exponentsAtOrderKTuple = combinations_with_replacement_counts(dim,k)
        exponentsAtOrderKTuple = list(exponentsAtOrderKTuple)
        listForm = [i[j] for i in exponentsAtOrderKTuple for j in range(len(i))]
        allExponents = allExponents + listForm
    separated = []
    for j in range(int(len(allExponents)/dim)):
        separated.append(allExponents[j*dim:dim*(j+1)])
    for count in range(len(separated)):
        basisElement = 1
        for count2 in range(len(separated[count])):
            basisElement = basisElement*mbedSymbols[count2]**separated[count][count2]
        basis.append(basisElement)
    return basis

class diffEqLinearModel():
    def __init__(self, ID = 'untitledModel'):
        self.ID = ID
        self.symbols = []
        self.basis = []
        self.Aij = []
        self.AijShort = []
        self.componentErrors = []
        self.longForm = None
        self.shortForm = None
        self.dt = 1
        self.error = None
        self.entropy = None
        self.lambdified = None

    def info(self):
        print('Model Info:')
        print('ID: ', end = '')
        print(self.ID)
        print('symbols: ', end = '')
        print(self.symbols)
        print('basis: ', end = '')
        print(self.basis)
        print()
        print('Short-Form Differential Equations of Motion: ')
        for i in range(len(self.shortFormDiffEq)):
            print('dot' + str(self.symbols[i])+' = ' + str(self.shortFormDiffEq[i]))
        print()
        print('Short-Form Map Equations: ')
        for i in range(len(self.shortForm)):
            print(str(self.symbols[i])+'_{t+1} = ' + str(self.shortForm[i]))
        print()
        print('Model Entropy: ')
        print(self.entropy)

    def save(self):
        pickle.dump(self,open('models/' + str(self.ID) + '_model.pickle','wb'))

    def fastFillFromDataset(self,reconstructedDataSet,basisType,order,printouts = False,shortformThresh = .01):
        self.symbols = [sympy.symbols(compname) for compname in reconstructedDataSet.componentNames]
        self.dt = reconstructedDataSet.dt
        self.basis = getBasis(basisType,order,self.symbols)
        self.Aij, self.componentErrors = self.leastSquaresAij(reconstructedDataSet,printouts)
        self.longForm = self.Aij.dot(self.basis)
        self.AijShort = self.removeSmallElements(shortformThresh)
        self.shortForm = self.AijShort.dot(self.basis)
        self.longFormDiffEq = self.getDiffEqFromMap(self.Aij)
        self.shortFormDiffEq = self.getDiffEqFromMap(self.AijShort)
        self.error = np.linalg.norm(self.componentErrors)
        self.entropy = numNonzeroElems(self.AijShort) + np.log(self.error)/np.log(2) #TODO maybe recalculate error here.
        self.lambdified = lambdify([tuple(self.symbols)],self.AijShort.dot(self.basis),'numpy')
        #I implicitly assume that the small terms don't contribute noticeably to the erorr.

    def removeSmallElements(self,threshold):
        allAij = self.Aij
        numEqn = len(allAij)
        shortFormAij = []
        for eqnCount in range(numEqn):
            row = []
            biggestInAij = max(abs(allAij[eqnCount]))
            for i in range(len(allAij[eqnCount])):
                if abs(allAij[eqnCount][i]) > abs(threshold * biggestInAij):
                    row.append(round_sig(allAij[eqnCount][i], 5))
                else:
                    row.append(0)
            shortFormAij.append(row)
        shortFormAij = np.array(shortFormAij)
        return shortFormAij

    def leastSquaresAij(self,reconstructedDataSet,printouts):
        allAij = []
        allSigs = []
        reconstructedData = reconstructedDataSet.compData
        timeReconstructedData = reconstructedDataSet.data
        nPoints = len(timeReconstructedData)
        basis = self.basis
        dimensionVars = self.symbols
        for componentCount in range(len(reconstructedData)):
            if printouts == True:
                print()
                print()
                print('Solving component ' + str(componentCount))
            startValMatrixA = np.array(np.zeros((nPoints - 1, len(basis))), dtype='float')
            outputVecB = np.array(reconstructedData[componentCount][1:], dtype='float')
            timeConstructStart = time()
            prevPercent = -1
            for j in range(len(basis)):
                func = lambdify([tuple(dimensionVars)], basis[j], 'numpy')
                percentDone = round(j / len(basis) * 100)
                if percentDone < prevPercent + 5:
                    pass
                else:
                    prevPercent = percentDone
                    if printouts == True:
                        print(str(j) + '/' + str(len(basis)), end='||')
                for i in range(nPoints - 1):
                    evaluated = float(func([float(elem) for elem in timeReconstructedData[
                        i]]))  # gross, has to be passed just the right kind of inputs (lists of float)
                    if type(evaluated) != list:
                        startValMatrixA[i, j] = evaluated
                    else:
                        print('hmmm something is up with evaluating the basis at ' + str(timeReconstructedData[i]))
                        print('type given as ' + str(type(evaluated)))
                        input()
                        startValMatrixA[i, j] = float(evaluated[0])
            timeConstructEnd = time()
            if printouts == True:
                print('Construction took ' + str(timeConstructEnd - timeConstructStart) + 'seconds.')
                print()
                print('Solving by least squares.')
            timeStart = time()
            bestAij = leastSquares(startValMatrixA, outputVecB, rcond=None)
            bestAij = bestAij[0]
            vecOfDifferences = (outputVecB - startValMatrixA.dot(bestAij))
            squaredVec = [elem ** 2 for elem in vecOfDifferences]
            sigi = np.sqrt(1 / nPoints * sum(squaredVec))
            allAij.append(bestAij)
            allSigs.append(sigi)
            timeFinish = time()
            if printouts == True:
                print('Solution complete. Elapsed time ' + str(timeFinish - timeStart) + ' seconds.')
        allAij = np.array(allAij)
        return allAij, allSigs

    def getDiffEqFromMap(self,Aij):
        allDerivatives = []
        symVars = self.symbols
        for k in range(len(symVars)):
            AijHere = Aij[k]
            aijCopy = list(abs(AijHere.copy()))
            aijCopy.sort()
            secondLargest = aijCopy[-2]
            biggerThanThis = .001 * secondLargest
            AijHereMod = []
            for i in range(len(AijHere)):
                if abs(AijHere[i]) < biggerThanThis:
                    AijHereMod.append(0)
                else:
                    AijHereMod.append(round_sig(AijHere[i], 5))
            funcHere = np.array(AijHereMod).dot(self.basis)
            deriv = (funcHere - symVars[k]) / self.dt
            allDerivatives.append(deriv)
        return allDerivatives


    def evolve(self,nPoints,initialConditions):
        allPoints = [initialConditions]
        for i in range(nPoints):
            allPoints.append(self.predict(allPoints[-1]))
        return allPoints

    def predict(self,point,tauFuture=1):
        currentPrediction = point
        for i in range(tauFuture):
            currentPrediction = self.lambdified(currentPrediction)
        return currentPrediction

class mapLinearModel():
    def __init__(self, ID = 'untitledModel'):
        self.ID = ID
        self.symbols = []
        self.basis = []
        self.Aij = []
        self.AijShort = []
        self.componentErrors = []
        self.longForm = None
        self.shortForm = None
        self.shortFormDiffEq = None
        self.longFormDiffEq = None
        self.dt = 1
        self.error = None
        self.entropy = None
        self.lambdified = None

    def info(self):
        print('Model Info:')
        print('ID: ', end = '')
        print(self.ID)
        print('symbols: ', end = '')
        print(self.symbols)
        print('basis: ', end = '')
        print(self.basis)
        print()
        print('Short-Form Differential Equations of Motion: ')
        for i in range(len(self.shortFormDiffEq)):
            print('dot' + str(self.symbols[i])+' = ' + str(self.shortFormDiffEq[i]))
        print()
        print('Short-Form Map Equations: ')
        for i in range(len(self.shortForm)):
            print(str(self.symbols[i])+'_{t+1} = ' + str(self.shortForm[i]))
        print()
        print('Model Entropy: ')
        print(self.entropy)

    def save(self):
        pickle.dump(self,open('models/' + str(self.ID) + '_model.pickle','wb'))

    def fastFillFromDataset(self,reconstructedDataSet,basisType,order,printouts = False,shortformThresh = .01):
        self.symbols = [sympy.symbols(compname) for compname in reconstructedDataSet.componentNames]
        self.dt = reconstructedDataSet.dt
        self.basis = getBasis(basisType,order,self.symbols)
        self.Aij, self.componentErrors = self.leastSquaresAij(reconstructedDataSet,printouts)
        self.longForm = self.Aij.dot(self.basis)
        self.AijShort = self.removeSmallElements(shortformThresh)
        self.shortForm = self.AijShort.dot(self.basis)
        self.longFormDiffEq = self.getDiffEqFromMap(self.Aij)
        self.shortFormDiffEq = self.getDiffEqFromMap(self.AijShort)
        self.error = np.linalg.norm(self.componentErrors)
        self.entropy = numNonzeroElems(self.AijShort) + np.log(self.error)/np.log(2) #TODO maybe recalculate error here.
        self.lambdified = lambdify([tuple(self.symbols)],self.AijShort.dot(self.basis),'numpy')
        #I implicitly assume that the small terms don't contribute noticeably to the erorr.

    def removeSmallElements(self,threshold):
        allAij = self.Aij
        numEqn = len(allAij)
        shortFormAij = []
        for eqnCount in range(numEqn):
            row = []
            biggestInAij = max(abs(allAij[eqnCount]))
            for i in range(len(allAij[eqnCount])):
                if abs(allAij[eqnCount][i]) > abs(threshold * biggestInAij):
                    row.append(round_sig(allAij[eqnCount][i], 5))
                else:
                    row.append(0)
            shortFormAij.append(row)
        shortFormAij = np.array(shortFormAij)
        return shortFormAij

    def leastSquaresAij(self,reconstructedDataSet,printouts):
        allAij = []
        allSigs = []
        reconstructedData = reconstructedDataSet.compData
        timeReconstructedData = reconstructedDataSet.data
        nPoints = len(timeReconstructedData)
        basis = self.basis
        dimensionVars = self.symbols
        for componentCount in range(len(reconstructedData)):
            if printouts == True:
                print()
                print()
                print('Solving component ' + str(componentCount))
            startValMatrixA = np.array(np.zeros((nPoints - 1, len(basis))), dtype='float')
            outputVecB = np.array(reconstructedData[componentCount][1:], dtype='float')
            timeConstructStart = time()
            prevPercent = -1
            for j in range(len(basis)):
                func = lambdify([tuple(dimensionVars)], basis[j], 'numpy')
                percentDone = round(j / len(basis) * 100)
                if percentDone < prevPercent + 5:
                    pass
                else:
                    prevPercent = percentDone
                    if printouts == True:
                        print(str(j) + '/' + str(len(basis)), end='||')
                for i in range(nPoints - 1):
                    evaluated = float(func([float(elem) for elem in timeReconstructedData[
                        i]]))  # gross, has to be passed just the right kind of inputs (lists of float)
                    if type(evaluated) != list:
                        startValMatrixA[i, j] = evaluated
                    else:
                        print('hmmm something is up with evaluating the basis at ' + str(timeReconstructedData[i]))
                        print('type given as ' + str(type(evaluated)))
                        input()
                        startValMatrixA[i, j] = float(evaluated[0])
            timeConstructEnd = time()
            if printouts == True:
                print('Construction took ' + str(timeConstructEnd - timeConstructStart) + 'seconds.')
                print()
                print('Solving by least squares.')
            timeStart = time()
            bestAij = leastSquares(startValMatrixA, outputVecB, rcond=None)
            bestAij = bestAij[0]
            vecOfDifferences = (outputVecB - startValMatrixA.dot(bestAij))
            squaredVec = [elem ** 2 for elem in vecOfDifferences]
            sigi = np.sqrt(1 / nPoints * sum(squaredVec))
            allAij.append(bestAij)
            allSigs.append(sigi)
            timeFinish = time()
            if printouts == True:
                print('Solution complete. Elapsed time ' + str(timeFinish - timeStart) + ' seconds.')
        allAij = np.array(allAij)
        return allAij, allSigs

    def getDiffEqFromMap(self,Aij):
        allDerivatives = []
        symVars = self.symbols
        for k in range(len(symVars)):
            AijHere = Aij[k]
            aijCopy = list(abs(AijHere.copy()))
            aijCopy.sort()
            secondLargest = aijCopy[-2]
            biggerThanThis = .001 * secondLargest
            AijHereMod = []
            for i in range(len(AijHere)):
                if abs(AijHere[i]) < biggerThanThis:
                    AijHereMod.append(0)
                else:
                    AijHereMod.append(round_sig(AijHere[i], 5))
            funcHere = np.array(AijHereMod).dot(self.basis)
            deriv = (funcHere - symVars[k]) / self.dt
            allDerivatives.append(deriv)
        return allDerivatives


    def evolve(self,nPoints,initialConditions):
        allPoints = [initialConditions]
        for i in range(nPoints):
            allPoints.append(self.predict(allPoints[-1]))
        return allPoints

    def predict(self,point,tauFuture=1):
        currentPrediction = point
        for i in range(tauFuture):
            currentPrediction = self.lambdified(currentPrediction)
        return currentPrediction

class experiment():
    def __init__(self,generationObj,datasetObj,modelObj):
        self.dataGenerator = generationObj
        self.dataset = datasetObj
        self.model = modelObj


'''
thisboi = loadDataset('Lorenz_1000')
lorenzModel = linearModel('LorenzModelTest')
lorenzModel.fastFillFromDataset(thisboi,'Taylor',2,True)
print(lorenzModel.shortFormDiffEq)
print(lorenzModel.predict([1,0,0],20000))
thisboi.plot([0,1,2])
'''