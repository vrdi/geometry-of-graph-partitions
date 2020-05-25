#imports
import operator
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def landmarkMDS_2D(parts, num_parts, distance_function, num_landmarks=3, plot=False, verbose=False, chooselandmarks=[]):
    '''
    Does landmark MDS with furthest point sampling.

    :param parts: a function (!) which takes an integer < num_parts and returns an object
    :param distance_function: a distance function of the form f(object1, object2)
    :param num_landmarks: number of landmark points
    :param plot: whether to plot the MDS

    :returns: x coordinates, y coordinates
    '''
    #generate landmarks
    landmarks = chooselandmarks.copy()
    if len(landmarks) == 0:
        lm = np.random.choice(num_parts)
        landmark = lm #first landmark point's index in parts
    else:
        landmark = chooselandmarks[0]
    landmarks = [landmark] #list of landmark indices
    distMat = np.zeros((num_parts, num_parts)) #incomplete distance matrix
    for index in range(num_landmarks-1):
        if verbose:
            print("\n Landmark #", index)
        max_distance = 0
        argmax_distance = landmark
        landmarkpart = parts(landmark)
        #compute distance to all other plans
        for j in range(num_parts):
            if verbose:
                print(j, end=" ")
            part = parts(j)
            d = distance_function(landmarkpart, parts(j))
            distMat[landmark,j] = d
            #determine whether min to landmarks is maximal so far
            to_lm = min([distMat[l,j] for l in landmarks])
            if (max_distance < to_lm) and (j not in landmarks):
                max_distance = to_lm
                argmax_distance = j
        #next landmark is furthest from landmarks
        if len(landmarks) >= len(chooselandmarks):
            landmark = argmax_distance
        else:
            landmark = chooselandmarks[len(landmarks)]
        landmarks.append(landmark)
    #fill in distances for last landmark
    landmarkpart = parts(landmark)
    if verbose:
        print("\n Landmark #", num_landmarks-1)
    for j in range(num_parts):
        if verbose:
            print(j, end=" ")
        part = parts(j)
        d = distance_function(landmarkpart, parts(j))
        distMat[landmark,j] = d

    #classical MDS on the landmarks
    lmdistMat = np.zeros((num_landmarks, num_landmarks))
    for i, landi in enumerate(landmarks):
        for j, landj in enumerate(landmarks):
            lmdistMat[i,j] = distMat[landi, landj]
    n = len(landmarks)
    sqdistMat = np.multiply(lmdistMat, lmdistMat)
    H = -1/n*np.ones((n,n))
    for i in range(n):
        H[i][i] += 1
    B = -0.5*np.dot(np.dot(H, sqdistMat), H)
    lambdas, vs = np.linalg.eig(B) #find eigenvalues
    pairs = sorted(
        [(lambdas[i],vs[:,i]) for i in range(len(lambdas))],
        key=operator.itemgetter(0),
        reverse=True
    ) #sort eigenvalues descending
    lmX = np.sqrt(pairs[0][0])*pairs[0][1]
    lmY = np.sqrt(pairs[1][0])*pairs[1][1]
    lmXdict = {landmarks[n]:lmX[n] for n in range(len(landmarks))} #maps index in parts to x coordinate
    lmYdict = {landmarks[n]:lmY[n] for n in range(len(landmarks))} #maps index in parts to y coordinate

    #set up parameters to embed the rest
    Deltan = np.zeros((len(landmarks), len(landmarks)))
    L = np.zeros((2, len(landmarks)))
    for i in range(len(landmarks)):
        L[0][i] = lmX[i]
        L[1][i] = lmY[i]
    Lsharp = L.copy()
    for i in range(2):
        Lsharp[i][:] /= np.linalg.norm(L[i][:])**2
    for i in range(len(landmarks)):
        for j in range(len(landmarks)):
            Deltan[i][j] = lmdistMat[i][j]**2
    Deltanbar = np.mean(Deltan, 1)

    #embed the rest
    colors = []
    xcoords = []
    ycoords = []
    Deltax_list = []
    for i in range(num_parts):
        if i not in landmarks:
            colors.append('b')
            part = parts(i)
            Deltax = np.zeros(len(landmarks))
            for j in range(len(landmarks)):
                lpartindex = landmarks[j]
                d = distMat[lpartindex, i]
                Deltax[j] = d**2
            Deltax_list.append(Deltax)
            x = 0.5*np.dot(Lsharp, Deltanbar-Deltax)
            xcoords.append(x[0])
            ycoords.append(x[1])
        else:
            colors.append('r')
            xcoords.append(lmXdict[i])
            ycoords.append(lmYdict[i])

    #plot
    if plot:
        ax = plt.axes()
        ax.scatter(xcoords, ycoords,color=colors)
        for i, txt in enumerate(landmarks):
            ax.annotate(txt, (xcoords[txt], ycoords[txt]))
    return xcoords, ycoords
