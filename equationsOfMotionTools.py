import numpy as np

def euler(lambdifiedDerivs,point,dt):
    numEqs = len(lambdifiedDerivs)
    pointOut = []
    for i in range(numEqs):
        pointOut.append(point[i] + lambdifiedDerivs[i](*point) * dt)
    return pointOut

def RK4Tangent(lambdifiedDerivs,point,dt): #The first len(lambdifiedDerivs) entries of point must be "static." (x, y, and z, FE, not dx, dy, dz)
    numEqs = len(lambdifiedDerivs)
    k1s = []
    k2s = []
    k3s = []
    k4s = []
    k2Args = point[0:numEqs]
    k3Args = point[0:numEqs]
    k4Args = point[0:numEqs]
    for i in range(numEqs):
        k1Here = dt * lambdifiedDerivs[i](*point)
        #input('fast k1' + str(k1Here))
        k1s.append(k1Here)
        k2Args.append(point[i+numEqs] + k1Here/2.0)
    for i in range(numEqs):
        k2Here = dt * lambdifiedDerivs[i](*k2Args)
        k2s.append(k2Here)
        k3Args.append(point[i+numEqs]+k2Here/2.0)
    for i in range(numEqs):
        k3Here = dt * lambdifiedDerivs[i](*k3Args)
        k3s.append(k3Here)
        k4Args.append(point[i+numEqs] + k3Here)
    for i in range(numEqs):
        k4Here = dt * lambdifiedDerivs[i](*k4Args)
        k4s.append(k4Here)

    pointOut = []
    for i in range(numEqs):
        pointOut.append( point[i+numEqs] + (k1s[i] + 2.0*k2s[i] + 2.0*k3s[i] + k4s[i]) / 6.0)
    return pointOut

def RK4(lambdifiedDerivs,point,dt):
    numEqs = len(lambdifiedDerivs)
    k1s = []
    k2s = []
    k3s = []
    k4s = []
    k2Args = []
    k3Args = []
    k4Args = []
    for i in range(numEqs):
        k1Here = dt * lambdifiedDerivs[i](*point)
        #'slow k1' + str(k1Here))
        k1s.append(k1Here)
        k2Args.append(point[i] + k1Here/2.0)
    for i in range(numEqs):
        k2Here = dt * lambdifiedDerivs[i](*k2Args)
        k2s.append(k2Here)
        k3Args.append(point[i]+k2Here/2.0)
    for i in range(numEqs):
        k3Here = dt * lambdifiedDerivs[i](*k3Args)
        k3s.append(k3Here)
        k4Args.append(point[i] + k3Here)
    for i in range(numEqs):
        k4Here = dt * lambdifiedDerivs[i](*k4Args)
        k4s.append(k4Here)

    pointOut = []
    for i in range(numEqs):
        pointOut.append( point[i] + (k1s[i] + 2.0*k2s[i] + 2.0*k3s[i] + k4s[i]) / 6.0)
    return pointOut





'''
    k1x = dt * f(a, b, c, x, y, z)
    k2x = dt * f(a, b, c, x + k1x / 2.0, y + k1y / 2.0, z + k1z / 2.0)
    k3x = dt * f(a, b, c, x + k2x / 2.0, y + k2y / 2.0, z + k2z / 2.0)
    k4x = dt * f(a, b, c, x + k3x, y + k3y, z + k3z)
    x += (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
'''