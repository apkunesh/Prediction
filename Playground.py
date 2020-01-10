from dataGeneration import *
import pickle
import sympy

lorenzSystem = loadDataGenerator('Lorenz_Typical')
longwindedLCEs = getLCEs(lorenzSystem,nTransients = 100,nIterates = 1000,slowOption = False)
print(longwindedLCEs)
pickle.dump(longwindedLCEs,open('lces.pickle','wb'))

'''
x,y,z,dx,dy,dz = sympy.symbols('x,y,z,dx,dy,dz')

xeqn = 10 * ( y -x )
yeqn = x * (28-z) - y
zeqn = x*y-8/3*z
allEqs = [xeqn,yeqn,zeqn]
symbols = [x,y,z]
ds = [dx,dy,dz]

linearizedEquations = []
for i in range(len(allEqs)):
    components = []
    for j in range(len(allEqs)):
        dEqHere = sympy.diff(allEqs[i],symbols[j])
        components.append(dEqHere * ds[j])
    linearizedEquations.append(sum(components))

print(linearizedEquations)
'''