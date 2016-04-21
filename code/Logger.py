# -*- coding: utf-8 -*-
#author: Krzysztof xaru Rajda


import time
import datetime

class Logger:

    def __init__(self, savePath, verbose):
        
        self.logFile = open(savePath+'dbn_training.log', 'a')    
        self.paramsFile = open(savePath+'parameters.log', 'a')    
        self.verbose = verbose
    
    def __del__(self):
        if self.logFile is not None:
            self.logFile.close()
        
        if self.paramsFile is not None:
            self.paramsFile.close()
    
    def log(self, text):
        
        if len(text) > 0:
        
            if self.logFile is not None:
                
                timeStr = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                saveText = timeStr+"\t"+text
                
                self.logFile.write(saveText)
                self.logFile.write("\n")
            
            if self.verbose:
                print text
    
    def logParameter(self, text, value):
        self.paramsFile.write(text+': '+str(value))
        self.paramsFile.write("\n")
        
        