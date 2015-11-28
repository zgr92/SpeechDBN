#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import csv
import theano
import theano.tensor as T


phonesStates = {'a#1':0, 'a#2':1, 'a#3':2, 'b#1':3, 'b#2':4, 'b#3':5, 'c#1':6, 'c#2':7, 'c#3':8, 'd#1':9, 'd#2':10, 'd#3':11, 'e#1':12, 'e#2':13, 'e#3':14, 'f#1':15, 'f#2':16, 'f#3':17, 'g#1':18, 'g#2':19, 'g#3':20, 'h#1':21, 'h#2':22, 'h#3':23, 'i#1':24, 'i#2':25, 'i#3':26, 'j#1':27, 'j#2':28, 'j#3':29, 'k#1':30, 'k#2':31, 'k#3':32, 'l#1':33, 'l#2':34, 'l#3':35, 'm#1':36, 'm#2':37, 'm#3':38, 'n#1':39, 'n#2':40, 'n#3':41, 'o#1':42, 'o#2':43, 'o#3':44, 'p#1':45, 'p#2':46, 'p#3':47, 'r#1':48, 'r#2':49, 'r#3':50, 's#1':51, 's#2':52, 's#3':53, 't#1':54, 't#2':55, 't#3':56, 'u#1':57, 'u#2':58, 'u#3':59, 'v#1':60, 'v#2':61, 'v#3':62, 'y#1':63, 'y#2':64, 'y#3':65, 'z#1':66, 'z#2':67, 'z#3':68, 'dź#1':69, 'dź#2':70, 'dź#3':71, 'dż#1':72, 'dż#2':73, 'dż#3':74, 'cz#1':75, 'cz#2':76, 'cz#3':77, 'dz#1':78, 'dz#2':79, 'dz#3':80, 'e~#1':81, 'e~#2':82, 'e~#3':83, 'g^#1':84, 'g^#2':85, 'g^#3':86, 'k^#1':87, 'k^#2':88, 'k^#3':89, 'nn#1':90, 'nn#2':91, 'nn#3':92, 'o~#1':93, 'o~#2':94, 'o~#3':95, 'sz#1':96, 'sz#2':97, 'sz#3':98, 'ś#1':99, 'ś#2':100, 'ś#3':101, 'ź#1':102, 'ź#2':103, 'ź#3':104, 'ł#1':105, 'ł#2':106, 'ł#3':107, 'ż#1':108, 'ż#2':109, 'ż#3':110, 'ć#1':111, 'ć#2':112, 'ć#3':113, 'ń#1':114, 'ń#2':115, 'ń#3':116, 'sil#1':117, 'sil#2':118, 'sil#3':119}


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
    valid_set_x = getMFCCFeatures('../data/speech/valid/merged.mfcc', n_frames)
    valid_set_y = getTargets('../data/speech/valid/merged.target', n_frames)
    train_set_x = getMFCCFeatures('../data/speech/train/merged.mfcc', n_frames)
    train_set_y = getTargets('../data/speech/train/merged.target', n_frames)
    
    test_set_x = getMFCCFeatures('../data/speech/valid/merged.mfcc', n_frames)
    test_set_y = getTargets('../data/speech/valid/merged.target', n_frames)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

