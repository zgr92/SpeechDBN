#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import csv
import theano
import theano.tensor as T


phonesStates = {"sil":0, "iy":1, "v":2, "ix":3, "n":4, "eh":5, "f":6, "zh":7, "t":8, "uh":9, "k":10, "w":11, "ax":12, "s":13, "ow":14, "d":15, "hh":16, "ae":17, "ch":18, "er":19, "aa":20, "r":21, "b":22, "m":23, "ng":24, "g":25, "ay":26, "th":27, "ax-h":28, "ey":29, "l":30, "dh":31, "p":32, "dx":33, "aw":34, "z":35, "jh":36, "uw":37, "oy":38, "y":39}


def normalizeZeroMeanUnitStdDev(x,pos,means,stdDevs):
    y = x-means[pos]
    y = y/stdDevs[pos]
    return y


def getStateById(stateId):
    for state, sid in phonesStates.items():
        if sid == stateId:
            return state
    return 'None'


def getOutputState(state, phonesStates):
    zeros = [0] * len(phonesStates)
    zeros[phonesStates[state]] = 1
    return zeros


def getOutputStateId(state, phonesStates):
    return phonesStates[state]


def getMFCCFeatures(path, frames):
    fileMFCC = open(path, 'r')
    dataMFCC = csv.reader(fileMFCC, delimiter=';')
    tableMFCC = [row for row in dataMFCC]

    for i in range(len(tableMFCC)):
        tableMFCC[i].pop(0)
        tableMFCC[i] = [ float(x) for x in tableMFCC[i] ]

    m = numpy.array(tableMFCC)

    means = m.mean(axis=0)
    stdDevs = m.std(axis=0)


    for i in range(len(tableMFCC)):
        for j in range (len(tableMFCC[i])):
            tableMFCC[i][j] = normalizeZeroMeanUnitStdDev(tableMFCC[i][j], j, means, stdDevs)

    finalTableMFCC = []
    for i in range(frames/2, len(tableMFCC)-frames/2):
        finalTableMFCC.append([])
        for j in range (i-frames/2,i+frames/2 + 1):
            finalTableMFCC[-1].extend(tableMFCC[j])

    inputData = numpy.array(finalTableMFCC, theano.config.floatX)

    return inputData


def getTargets(path, frames):
    fileTarget = open(path, 'rb')
    dataTarget = csv.reader(fileTarget, delimiter=';')
    tableTarget = [row for row in dataTarget]

    for i in range(len(tableTarget)):
        tableTarget[i].pop(0)
        tableTarget[i] =  [ getOutputStateId(x, phonesStates) for x in tableTarget[i] ]
    
    tableTarget = [val for subl in tableTarget for val in subl]
    
    finalTableTarget = tableTarget[frames/2 : len(tableTarget)-frames/2]
    
    targetData = numpy.array(finalTableTarget)

    return targetData


def load_data(n_frames=11):
    valid_set_x = getMFCCFeatures('../data/TIMIT/valid/merged.mfcc', n_frames)
    valid_set_y = getTargets('../data/TIMIT/valid/merged.target', n_frames)

    train_set_x = getMFCCFeatures('../data/TIMIT/train/merged.mfcc', n_frames)
    train_set_y = getTargets('../data/TIMIT/train/merged.target', n_frames)
    
    test_set_x = getMFCCFeatures('../data/TIMIT/test/merged.mfcc', n_frames)
    test_set_y = getTargets('../data/TIMIT/test/merged.target', n_frames)

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

