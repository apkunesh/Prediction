import pickle
import sympy
from sympy.parsing import sympy_parser
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from equationsOfMotionTools import RK4, euler, RK4Tangent
from time import time


def loadDataGenerator(genNameStr):
    dataGeneratorToReturn = pickle.load(open('datagens/' + genNameStr + '_Generator.pickle','rb'))
    dataGeneratorToReturn.lambdified = [sympy.lambdify(dataGeneratorToReturn.symbols,equation) for equation in dataGeneratorToReturn.equations]
    return dataGeneratorToReturn

def loadDataset(datNameStr):
    return pickle.load(open('datasets/' + datNameStr + '.pickle','rb'))

class dataset():
    def __init__(self,dataIn = None, ID = 'untitledData', componentNames = None, color = 'red',dt = None):
        self.ID = ID
        self.data = dataIn
        self.compData = self.switchDataIndices(dataIn)
        self.dt = dt
        self.bestKnownModel = None
        self.additionalInfo = {}
        self.componentNames = []
        self.color = color
        self.spectrum  = np.fft.fftn(self.data) #TODO: make sure Spectrum performs as desired.
        if componentNames == None:
            self.componentNames = ['x'+str(k) for k in range(len(dataIn[0]))]
        else:
            self.componentNames = [str(nameboi) for nameboi in componentNames]

    def plot(self, componentList = None,showMe = True):
        if len(componentList) > 3:
            print('Too many dimensions to plot! Must be less than 4.')
        else:
            dataToPlot = []
            if type(componentList[0]) is int:
                for i in range(len(componentList)):
                    dataToPlot.append(self.compData[componentList[i]])
            elif type(componentList[0]) is str:
                for i in range(len(componentList)):
                    dataToPlot.append(self.compData[self.componentNames.index(componentList[i])])
                    componentList[i] = self.componentNames.index(componentList[i])
            if len(dataToPlot) == 1:
                plt.plot(dataToPlot[0],color = self.color,marker='*',linestyle = '')
                plt.ylabel(self.componentNames[componentList[0]])
                plt.xlabel('index')
                plt.title(str(self.ID))
            elif len(dataToPlot) == 2:
                plt.plot(dataToPlot[0],dataToPlot[1],color = self.color,marker='*',linestyle = '')
                plt.xlabel(self.componentNames[componentList[0]])
                plt.ylabel(self.componentNames[componentList[1]])
                plt.title(str(self.ID))
            elif len(dataToPlot) == 3:
                ax = plt.axes(projection='3d')
                ax.scatter3D(dataToPlot[0],dataToPlot[1],dataToPlot[2], color = self.color)
                ax.set_xlabel(str(self.componentNames[componentList[0]]))
                ax.set_ylabel(str(self.componentNames[componentList[1]]))
                ax.set_zlabel(str(self.componentNames[componentList[2]]))
            if showMe == True:
                plt.show()

    def switchDataIndices(self,whichData):
        dataByTime = []
        for i in range(len(whichData[0])):
            rowhere = []
            for j in range(len(whichData)):
                rowhere.append(whichData[j][i])
            dataByTime.append(rowhere)
        return dataByTime

    def info(self):
        print('Dataset Information: ')
        print('ID: ' + str(self.ID))
        print('Component names: ' + str([component for component in self.componentNames]))
        print('Length of series: ' + str(len(self.data)))
        if self.dt is not None: print('dt = ' + str(self.dt))
        if self.bestKnownModel is not None: print('bestModel: ' + str(self.bestKnownModel.ID))
        if self.additionalInfo is not None: print('Additional Info: ' + str(self.additionalInfo))
        print()

    def save(self,fileName = None):
        if fileName is None:
            pickle.dump(self,open('datasets/' + str(self.ID) + '.pickle','wb'))
        else:
            pickle.dump(self,open('datasets/'+fileName+'.pickle','wb'))

class mapDataGenerator():
    def __init__(self, ID = 'untitledMap'):
        self.ID = ID
        self.equations = []
        self.symbols = []
        self.lambdified = []
        self.trueGenerator = None
    def printEquations(self):
        for i in range(len(self.equations)):
            print(str(self.symbols[i]) + '_next = ' + str(self.equations[i]))
    def printSymbols(self):
        for i in range(len(self.equations)):
            print(self.symbols[i], end=' ')
        print()
    def info(self):
        print('Equation Information: ')
        print('ID: ' + str(self.ID))
        print('Equations: ')
        self.printEquations()
        print('Sympy symbols:' )
        self.printSymbols()
        print()
    def save(self,filename = None):
        self.lambdified = []
        generatorHolder = self.trueGenerator
        self.trueGenerator = None
        if filename is None:
            pickle.dump(self,open('datagens/' + str(self.ID) + '_MapGenerator.pickle','wb'))
        else:
            pickle.dump(self,open('datagens/' + filename + '_MapGenerator.pickle','wb'))
        self.trueGenrator = generatorHolder
        self.lambdified = [sympy.lambdify(self.symbols,equation) for equation in self.equations]
    def evolve(self,nPoints,initialConditions = []):
        if initialConditions == [] and self.trueGenerator == None:
            print('Error: When first evolving the system, you must specify initialConditions')
            return
        elif initialConditions != [] and self.trueGenerator == None:
            self.trueGenerator = self.makeGenerator(initialConditions)
            allPoints = [initialConditions]
        else:
            allPoints = []
        for i in range(nPoints):
            allPoints.append(next(self.trueGenerator))
        return allPoints
    def setSymbols(self,listOrInt):
        if type(listOrInt) == int:
            nSymbols = listOrInt
            for i in range(nSymbols):
                self.symbols.append(sympy.symbols('x'+str(i)))
        elif type(listOrInt) ==  list:
            listOfSymbolNames = listOrInt
            for i in range(len(listOfSymbolNames)):
                self.symbols.append(sympy.symbols(listOfSymbolNames[i]))
    def setEquations(self,equationStringList):
        self.equations = []
        for i in range(len(equationStringList)):
            self.equations.append(sympy_parser.parse_expr(equationStringList[i]))
        self.lambdified = [sympy.lambdify(self.symbols,equation) for equation in self.equations]

    def renameSymbols(self,newNamesList):
        if len(newNamesList) != len(self.symbols): print('ERROR, wrong # names given for symbols.')
        translateDict = {}
        for i in range(len(newNamesList)):
            oldSymbols = self.symbols.copy()
            self.symbols[i] = sympy.symbols(newNamesList[i])
            translateDict[oldSymbols[i]] = self.symbols[i]
        for i in range(len(self.equations)):
            self.equations[i] = self.equations[i].subs(translateDict)
    def makeGenerator(self,initialConditions):
        inputs  = [inputval for inputval in initialConditions]
        while True:
            tempInputs = []
            for j in range(len(self.lambdified)):
                tempInputs.append(self.lambdified[j](*inputs))
            inputs = tempInputs
            yield inputs

class diffEqDataGenerator():
    def __init__(self, ID = 'untitledEquation'):
        self.ID = ID
        self.dt = .001
        self.equations = []
        self.symbols = []
        self.lambdified = []
        self.trueGenerator = None
    def printEquations(self):
        for i in range(len(self.equations)):
            print('/dot ' + str(self.symbols[i]) + ' = ' + str(self.equations[i]))
    def printSymbols(self):
        for i in range(len(self.equations)):
            print(self.symbols[i], end=' ')
        print()
    def info(self):
        print('Equation Information: ')
        print('ID: ' + str(self.ID))
        print('Equations: ')
        self.printEquations()
        print('Sympy symbols:' )
        self.printSymbols()
        print()
    def save(self,filename = None):
        self.lambdified = []
        generatorHolder = self.trueGenerator
        self.trueGenerator = None
        if filename is None:
            pickle.dump(self,open('datagens/' + str(self.ID) + '_Generator.pickle','wb'))
        else:
            pickle.dump(self,open('datagens/' + filename + '_Generator.pickle','wb'))
        self.lambdified = [sympy.lambdify(self.symbols,equation) for equation in self.equations]
        self.trueGenerator = generatorHolder
    def evolve(self,nPoints,initialConditions = [],integrationMethod = RK4): #TODO: Integrate modular integration here.
        if initialConditions == [] and self.trueGenerator == None:
            print('Error: When first evolving the system, you must specify initialConditions')
            return
        elif initialConditions != [] and self.trueGenerator == None:
            self.trueGenerator = self.makeGenerator(initialConditions, integrationMethod)
            allPoints = [initialConditions]
        else:
            allPoints = []
        for i in range(nPoints):
            allPoints.append(next(self.trueGenerator))
        return allPoints
    def setSymbols(self,listOrInt):
        if type(listOrInt) == int:
            nSymbols = listOrInt
            for i in range(nSymbols):
                self.symbols.append(sympy.symbols('x'+str(i)))
        elif type(listOrInt) ==  list:
            listOfSymbolNames = listOrInt
            for i in range(len(listOfSymbolNames)):
                self.symbols.append(sympy.symbols(listOfSymbolNames[i]))
    def setEquations(self,equationStringList):
        self.equations = []
        for i in range(len(equationStringList)):
            self.equations.append(sympy_parser.parse_expr(equationStringList[i]))
        self.lambdified = [sympy.lambdify(self.symbols,equation) for equation in self.equations]
    def renameSymbols(self,newNamesList):
        if len(newNamesList) != len(self.symbols): print('ERROR, wrong # names given for symbols.')
        translateDict = {}
        for i in range(len(newNamesList)):
            oldSymbols = self.symbols.copy()
            self.symbols[i] = sympy.symbols(newNamesList[i])
            translateDict[oldSymbols[i]] = self.symbols[i]
        for i in range(len(self.equations)):
            self.equations[i] = self.equations[i].subs(translateDict)
    def makeGenerator(self,initialConditions,integrationMethod = RK4):
        inputs = [inputval for inputval in initialConditions]
        while True:
            inputs = integrationMethod(self.lambdified, inputs, self.dt)
            yield(inputs)



def getLCEs(modelOrDataGenerator,initialConditions = [], nTransients = 1000, nIterates = 10000,nItsPerPB = 10,slowOption = False,statusUpdates = True):
    numComps = len(modelOrDataGenerator.symbols)
    tanVecs = []
    derivatives = []
    #establish tangent vectors
    for i in range(numComps):
        vecHere = [0] * numComps
        vecHere[i] = 1
        tanVecs.append(vecHere)
    #establish linearized tangent-space equations
    linearizedEquations = []
    dsyms = [sympy.symbols('d' + str(symbolboi)) for symbolboi in modelOrDataGenerator.symbols]
    for i in range(numComps):
        components = []
        for j in range(numComps):
            dEqHere = sympy.diff(modelOrDataGenerator.equations[i], modelOrDataGenerator.symbols[j])
            components.append(dEqHere * dsyms[j])
        linearizedEquations.append(sum(components))
    lambdifiedNonlocalLEs = [sympy.lambdify(modelOrDataGenerator.symbols + dsyms,equationBoi) for equationBoi in linearizedEquations]

    # Iterate away transients and let the tangent vectors align
    #	with the global stable and unstable manifolds
    if initialConditions == []:
        initialConditions = [1 for i in modelOrDataGenerator.symbols]
    modelOrDataGenerator.trueGenerator = modelOrDataGenerator.makeGenerator(initialConditions)
    prevTime = 0
    print('Computing Transients')
    for n in range(0, nTransients):
        if statusUpdates == True:
            if time()-5 > prevTime:
                print(str(n/nTransients*100) + '%   ', end = ' ')
                prevTime = time()
        for i in range(nItsPerPB):
            currentState = next(modelOrDataGenerator.trueGenerator)
            if slowOption == True:
                subsDict = [(modelOrDataGenerator.symbols[index1],currentState[index1]) for index1 in range(numComps)]
                localLinearizedEquations = [kappa.subs(subsDict) for kappa in linearizedEquations]
                lambdifiedLLEs = [sympy.lambdify(dsyms,LLE) for LLE in localLinearizedEquations] #TODO get ride of this multi-loop lambdification if possible. Probably involves altering RK4 or making a new one for tangent space.
                for j in range(len(tanVecs)):
                    #print('computing Tangent')
                    tanVecs[j] = RK4(lambdifiedLLEs,tanVecs[j],modelOrDataGenerator.dt)
                #input('slowVecs' + str(tanVecs))

            else:
                for j in range(len(tanVecs)):
                    #input(currentState + tanVecs[j])
                    #print('computing Tangent')
                    tanVecs[j] = RK4Tangent(lambdifiedNonlocalLEs,currentState+tanVecs[j],modelOrDataGenerator.dt)
                #input('fastVecs' + str(tanVecs))

        #Pullback phase
        for i in range(len(tanVecs)-1):
            normHere = np.linalg.norm(tanVecs[i])
            tanVecs[i] = [vecVal/normHere for vecVal in tanVecs[i]]
            allToBeSubtracted = [0]*numComps
            for j in range(i+1):
                currentProjection = sum(zippedVecs[0] * zippedVecs[1] for zippedVecs in zip(tanVecs[j], tanVecs[i+1]))
                allToBeSubtracted = [goober[0] + currentProjection * goober[1] for goober in zip(allToBeSubtracted,tanVecs[j])] #TODO FIX THIS GRAHAM-SCHMIDT!!
            tanVecs[i+1] = [goober[0] - goober[1] for goober in zip(tanVecs[i+1],allToBeSubtracted)]
        lastNorm = np.linalg.norm(tanVecs[-1])
        tanVecs[-1] = [vecVal/lastNorm for vecVal in tanVecs[-1]]
    print('100%')
    # Okay, now we're ready to begin the estimation
    LCEs = [0]*numComps
    prevTime = 0
    print('Computing LCEs')
    for n in range(0, nIterates):
        if statusUpdates == True:
            if time()-5 > prevTime:
                print(str(n/nIterates*100) + '%   ', end = ' ')
                prevTime = time()
        for i in range(nItsPerPB):
            currentState = next(modelOrDataGenerator.trueGenerator)
            if slowOption == True:
                subsDict = [(modelOrDataGenerator.symbols[index1], currentState[index1]) for index1 in range(numComps)]
                localLinearizedEquations = [kappa.subs(subsDict) for kappa in linearizedEquations]
                lambdifiedLLEs = [sympy.lambdify(dsyms, LLE) for LLE in
                                  localLinearizedEquations]  # TODO get ridof this multi-loop lambdification if possible. Probably involves altering RK4 or making a new one for tangent space.
                for j in range(len(tanVecs)):
                    tanVecs[j] = RK4(lambdifiedLLEs, tanVecs[j], modelOrDataGenerator.dt)
            else:
                for j in range(len(tanVecs)):
                    tanVecs[j] = RK4Tangent(lambdifiedNonlocalLEs,currentState+tanVecs[j],modelOrDataGenerator.dt)

        for i in range(len(tanVecs) - 1):
            normHere = np.linalg.norm(tanVecs[i])
            tanVecs[i] = [vecVal / normHere for vecVal in tanVecs[i]]
            LCEs[i] += np.log(normHere)
            allToBeSubtracted = [0]*numComps
            for j in range(i+1):
                currentProjection = sum(zippedVecs[0] * zippedVecs[1] for zippedVecs in zip(tanVecs[j], tanVecs[i+1]))
                allToBeSubtracted = [goober[0] + currentProjection * goober[1] for goober in zip(allToBeSubtracted,tanVecs[j])] #TODO FIX THIS GRAHAM-SCHMIDT!!
            tanVecs[i+1] = [goober[0] - goober[1] for goober in zip(tanVecs[i+1],allToBeSubtracted)]
        lastNorm = np.linalg.norm(tanVecs[-1])
        tanVecs[-1] = [vecVal / lastNorm for vecVal in tanVecs[-1]]
        LCEs[-1] += np.log(lastNorm)
    print('100%')

    IntegrationTime = modelOrDataGenerator.dt * float(nItsPerPB) * float(nIterates)
    LCEs = [exponentBoi/IntegrationTime for exponentBoi in LCEs]
    return LCEs
        # Normalize the tangent vector



'''
henonGen = mapDataGenerator('Henon_Typical')
henonGen.setSymbols(['x','y'])
henonGen.setEquations(['1 - 1.4*x**2 + y','.3*x'])
#henonGen.save()
print(henonGen.evolve(3,[1,1]))
'''

'''
thisboi = mapDataGenerator('kappa')
thisboi.setSymbols(['x','y'])
thisboi.setEquations(['3*x','2*y'])
print(thisboi.evolve(12,[1,1]))
'''