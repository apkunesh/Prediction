import pickle
import sympy
from sympy.parsing import sympy_parser
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import axes3d


def loadDataGenerator(genNameStr):
    return pickle.load(open('datagens/' + genNameStr + '_Generator.pickle','rb'))

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



class equationBasedDataGenerator():
    def __init__(self, ID = 'untitledEquation'):
        self.ID = ID
        self.equations = []
        self.dt = 1
        self.diffEquations = []
        self.symbols = []


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
        if filename is None:
            pickle.dump(self,open('datagens/' + str(self.ID) + '_Generator.pickle','wb'))
        else:
            pickle.dump(self,open('datagens/' + filename + '_Generator.pickle','wb'))
    def evolve(self,nPoints,initialConditions):
        allPoints = [initialConditions]
        lambdifiedEquations = [sympy.lambdify(self.symbols,equation) for equation in self.equations]
        for i in range(nPoints):
            nextPoint = []
            for j in range(len(lambdifiedEquations)):
                newestInputs = [inputval for inputval in allPoints[-1]]
                nextPoint.append(lambdifiedEquations[j](*newestInputs))
            allPoints.append(nextPoint)
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

    def setDiffEquations(self,equationStringList,dt):
        self.dt = dt
        self.diffEquations = []
        self.equations = []
        for i in range(len(equationStringList)):
            self.diffEquations.append(sympy_parser.parse_expr(equationStringList[i]))
            self.equations.append(self.symbols[i]+self.dt * self.diffEquations[i])


    def renameSymbols(self,newNamesList):
        if len(newNamesList) != len(self.symbols): print('ERROR, wrong # names given for symbols.')
        translateDict = {}
        for i in range(len(newNamesList)):
            oldSymbols = self.symbols.copy()
            self.symbols[i] = sympy.symbols(newNamesList[i])
            translateDict[oldSymbols[i]] = self.symbols[i]
        for i in range(len(self.equations)):
            self.equations[i] = self.equations[i].subs(translateDict)
            if self.diffEquations != []:
                self.diffEquations[i] = self.diffEquations[i].subs(translateDict)
