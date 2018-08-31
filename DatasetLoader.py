#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Dataset Loader
"""


import os
import string
import random
import codecs
import pickle
import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import StratifiedKFold, GroupKFold

from sklearn.datasets.base import Bunch
from sklearn.utils import shuffle

from textblob import TextBlob as tb

# Paths
currentPath= os.path.dirname(os.path.abspath(__file__))

class DatasetLoader:
    
    def __init__(self, corpusPath= os.path.abspath("data/generatedDatasets/en/essays/"), corpusFilename= "ArgEssaysCorpus_en"):
        
        self.corpusPath= corpusPath
        
        self.corpusFilename= corpusFilename
        
        # dataset
        self.dataset= Bunch()
        
        # dataset fields
        
        # each training data instance is dictionary with the following format: {"SourceADU": sourceADU_string, "TargetADU": targetADU_string}
        (self.dataset).data= [] 
        
        # 0 -> "None" (not related); 1 -> "Supports" (Argumentative relations -> valid premise-conclusion pair)
        (self.dataset).target= []
        
        (self.dataset).target_names= ["None", "Support"]
        
    
    def loadDataFromCSV(self):
        
        # load csv file
        corpus= pd.read_csv(self.corpusPath + "/" + self.corpusFilename + ".csv", sep= "\t")
        
        self.trainingSet= Bunch()
        (self.trainingSet).data= [] 
        (self.trainingSet).target= [] 
        (self.trainingSet).target_names= ["None", "Support"]
        
        for index, row in corpus[corpus["Partition"] == "TRAIN"].iterrows():
            ((self.trainingSet).data).append({
                "SourceADU": row.SourceADU, 
                "TargetADU": row.TargetADU, 
                "SourceADU_tokens": self.myTokenizer(row.SourceADU, lowercase= True, removePunctuationMarks= False),
                "TargetADU_tokens": self.myTokenizer(row.TargetADU, lowercase= True, removePunctuationMarks= False),
                "ArticleId": row.ArticleId
                })
            
            if row.RelationType == "supports":
                ((self.trainingSet).target).append(1)
            else:
                #NOTE: "none" and "attacks" in csv file are considered "None" here
                ((self.trainingSet).target).append(0)
        
        self.validationSet= Bunch()
        (self.validationSet).data= [] 
        (self.validationSet).target= [] 
        (self.validationSet).target_names= ["None", "Support"]
        
        for index, row in corpus[corpus["Partition"] == "VALIDATION"].iterrows():
            ((self.validationSet).data).append({
                "SourceADU": row.SourceADU, 
                "TargetADU": row.TargetADU, 
                "SourceADU_tokens": self.myTokenizer(row.SourceADU, lowercase= True, removePunctuationMarks= False),
                "TargetADU_tokens": self.myTokenizer(row.TargetADU, lowercase= True, removePunctuationMarks= False),
                "ArticleId": row.ArticleId
                })
            
            if row.RelationType == "supports":
                ((self.validationSet).target).append(1)
            else:
                #NOTE: "none" and "attacks" in csv file are considered "None" here
                ((self.validationSet).target).append(0)
        
        self.testSet= Bunch()
        (self.testSet).data= [] 
        (self.testSet).target= [] 
        (self.testSet).target_names= ["None", "Support"]
        
        for index, row in corpus[corpus["Partition"] == "TEST"].iterrows():
            ((self.testSet).data).append({
                "SourceADU": row.SourceADU, 
                "TargetADU": row.TargetADU, 
                "SourceADU_tokens": self.myTokenizer(row.SourceADU, lowercase= True, removePunctuationMarks= False),
                "TargetADU_tokens": self.myTokenizer(row.TargetADU, lowercase= True, removePunctuationMarks= False),
                "ArticleId": row.ArticleId
                })
            
            if row.RelationType == "supports":
                ((self.testSet).target).append(1)
            else:
                #NOTE: "none" and "attacks" in csv file are considered "None" here
                ((self.testSet).target).append(0)
        
        # Convert to numpy arrays
        
        (self.trainingSet).data= np.asarray((self.trainingSet).data)
        (self.trainingSet).target= np.asarray((self.trainingSet).target)
        
        (self.validationSet).data= np.asarray((self.validationSet).data) 
        (self.validationSet).target= np.asarray((self.validationSet).target)
        
        (self.testSet).data= np.asarray((self.testSet).data) 
        (self.testSet).target= np.asarray((self.testSet).target)
        
    
    
    def getCrossValidationSplits(self, completeDataset, nSplits):
        
        datasetUniqueArticlesIds= list(set([elem["ArticleId"] for elem in completeDataset.data]))
        
        articleIdUniqueIntKeyDict= {}
        integerKey= 0
        
        for articleId in datasetUniqueArticlesIds:
            articleIdUniqueIntKeyDict[articleId] = integerKey
            integerKey = integerKey + 1
        
        trainingSetGroups= np.asarray([articleIdUniqueIntKeyDict[elem["ArticleId"]] for elem in completeDataset.data])
        
        cvStrategyTrainingTestData= GroupKFold(n_splits= nSplits)
        cvFoldsTrainingTestData= cvStrategyTrainingTestData.split(completeDataset.data, completeDataset.target, trainingSetGroups)
        
        foldsPartition= []
        foldsPartitionIndexes= []
        
        for currentFoldTrainingSetIdx, currentFoldTestIdx in cvFoldsTrainingTestData:
            
            # get Training/Dev set partitions
            cvStrategyTrainingDevData= GroupKFold(n_splits= nSplits)
            cvFoldsTrainingDevData= cvStrategyTrainingDevData.split(completeDataset.data[currentFoldTrainingSetIdx], completeDataset.target[currentFoldTrainingSetIdx], trainingSetGroups[currentFoldTrainingSetIdx])
            
            cvFoldsSplits= [(trainingSetIdx, devSetIdx) for trainingSetIdx, devSetIdx in cvFoldsTrainingDevData]
            
            currentFoldTrainingSetFinalIdx= currentFoldTrainingSetIdx[cvFoldsSplits[0][0]]
            currentFoldDevIdx= currentFoldTrainingSetIdx[cvFoldsSplits[0][1]]
            
            currentFoldTrainingSet= Bunch()
            currentFoldTrainingSet.data= completeDataset.data[currentFoldTrainingSetFinalIdx]
            currentFoldTrainingSet.target= completeDataset.target[currentFoldTrainingSetFinalIdx]
            currentFoldTrainingSet.target_names= completeDataset.target_names
            
            
            currentFoldDevSet= Bunch()
            currentFoldDevSet.data= completeDataset.data[currentFoldDevIdx]
            currentFoldDevSet.target= completeDataset.target[currentFoldDevIdx]
            currentFoldDevSet.target_names= completeDataset.target_names
            
            
            currentFoldTestSet= Bunch()
            currentFoldTestSet.data= completeDataset.data[currentFoldTestIdx]
            currentFoldTestSet.target= completeDataset.target[currentFoldTestIdx]
            currentFoldTestSet.target_names= completeDataset.target_names
            
            foldsPartition.append((currentFoldTrainingSet, currentFoldDevSet, currentFoldTestSet))
            foldsPartitionIndexes.append((currentFoldTrainingSetFinalIdx, currentFoldDevIdx, currentFoldTestIdx))
        
        # store training, development and test set on pickle file
        cvFoldsPartitionInfoFile= codecs.open(str(self.corpusPath) + "/" + "FoldsPartition" + "/" + str(self.corpusFilename) + "_DatasetPartition_" + str(nSplits) + "Folds.pkl", mode= "w", encoding= "utf-8")
        
        pickle.dump(foldsPartition, cvFoldsPartitionInfoFile)
        
        cvFoldsPartitionInfoFile.close()
        
        # store learning instances indexes for training, development and test set
        cvFoldsPartitionIndexesInfoFile= codecs.open(str(self.corpusPath) + "/" + "FoldsPartition" + "/" + str(self.corpusFilename) + "_DatasetPartition_" + str(nSplits) + "FoldsIndexes.pkl", mode= "w", encoding= "utf-8")
        
        pickle.dump(foldsPartitionIndexes, cvFoldsPartitionIndexesInfoFile)
        
        cvFoldsPartitionIndexesInfoFile.close()
        
        return True
        
        
    
    
    def myTokenizer(self, text, lowercase= True, removePunctuationMarks= False):
        
        # Punctuation Marks list
        punctuationMarksList= set(string.punctuation)
        
        tokenizerRegularExpression = ur'''(?ux)
        # the order of the patterns is important!!
        (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
        \d+(?:[.,]\d+)*(?:[.,]\d+)|       # numbers in format 999.999.999,99999
        \w+(?:\.(?!\.|$))?|               # words with numbers (including hours as 12h30),
                                          # followed by a single dot but not at the end of sentence
        \d+(?:[-\\/]\d+)*|                # dates. 12/03/2012 12-03-2012
        \$|                               # currency sign
        -+|                               # any sequence of dashes
        \S                                # any non-space character
        '''
        tokenizer = RegexpTokenizer(tokenizerRegularExpression)
        
        tokenizedSentence= tokenizer.tokenize(text)
        
        if lowercase:
            tokenizedSentence= [tok.lower() for tok in tokenizedSentence]
        
        if removePunctuationMarks:
            tokenizedSentence= [tok for tok in tokenizedSentence if tok not in punctuationMarksList]
        
        return tokenizedSentence
    
    def randomUndersamplingForBinaryClassification(self, dataset, randomSeed= 12345):
        b2= Bunch()
        
        b2.data= []
        b2.target= []
        b2.target_names= dataset.target_names
        
        datasetTemp= Bunch()
        datasetTemp.data= []
        datasetTemp.target= []
        
        # shuffle dataset
        datasetTemp.data, datasetTemp.target= shuffle(dataset.data, dataset.target, random_state= randomSeed)
        
        
        positiveExamplesIndexes= [i for i in xrange(len(datasetTemp.target)) if datasetTemp.target[i] == 1]
        
        numberOfPositiveExamples= len(positiveExamplesIndexes)
        numberOfNegativeExamples= len(datasetTemp.target) - len(positiveExamplesIndexes)
        
        numberOfPositiveExamplesToKeep= 0
        numberOfNegativeExamplesToKeep= 0
        
        if numberOfPositiveExamples > numberOfNegativeExamples:
            numberOfPositiveExamplesToKeep= numberOfNegativeExamples
            numberOfNegativeExamplesToKeep= numberOfNegativeExamples
        else:
            numberOfNegativeExamplesToKeep= numberOfPositiveExamples
            numberOfPositiveExamplesToKeep= numberOfPositiveExamples
        
        for trainingExampleIndex in xrange(len(datasetTemp.target)):
            
            if datasetTemp.target[trainingExampleIndex] == 1:
                if numberOfPositiveExamplesToKeep > 0:
                    (b2.data).append(datasetTemp.data[trainingExampleIndex])
                    (b2.target).append(1)
                    numberOfPositiveExamplesToKeep -= 1
            else:
                if numberOfNegativeExamplesToKeep > 0:
                    (b2.data).append(datasetTemp.data[trainingExampleIndex])
                    (b2.target).append(0)
                    numberOfNegativeExamplesToKeep -= 1
            
        return b2
    
    def translateDataset(self, sourceLanguage, targetLanguage):
        
        print "\n Translating training set ..."
        
        trainingSetTranslations= Bunch()
        trainingSetTranslations.data= []
        trainingSetTranslations.target= []
        trainingSetTranslations.target_names= ["None", "Support"]
        
        # training set
        for learningInstanceIndex in xrange(len((self.trainingSet).target)):
            
            sourceADUContent= str(tb((self.trainingSet).data[learningInstanceIndex]["SourceADU"]).translate(from_lang= sourceLanguage, to= targetLanguage))
            targetADUContent= str(tb((self.trainingSet).data[learningInstanceIndex]["TargetADU"]).translate(from_lang= sourceLanguage, to= targetLanguage))
            
            (trainingSetTranslations.data).append({
                "SourceADU": sourceADUContent, 
                "TargetADU": targetADUContent,
                "SourceADU_tokens": self.myTokenizer(sourceADUContent, lowercase= True, removePunctuationMarks= False),
                "TargetADU_tokens": self.myTokenizer(targetADUContent, lowercase= True, removePunctuationMarks= False),
                "ArticleId": (self.trainingSet).data[learningInstanceIndex]["ArticleId"]
                })
            
            (trainingSetTranslations.target).append((self.trainingSet).target[learningInstanceIndex])
            
        
        print "\n Translating validation set ..."
        
        validationSetTranslations= Bunch()
        validationSetTranslations.data= []
        validationSetTranslations.target= []
        validationSetTranslations.target_names= ["None", "Support"]
        
        # validation set
        for learningInstanceIndex in xrange(len((self.validationSet).target)):
            
            sourceADUContent= str(tb((self.validationSet).data[learningInstanceIndex]["SourceADU"]).translate(from_lang= sourceLanguage, to= targetLanguage))
            targetADUContent= str(tb((self.validationSet).data[learningInstanceIndex]["TargetADU"]).translate(from_lang= sourceLanguage, to= targetLanguage))
            
            (validationSetTranslations.data).append({
                "SourceADU": sourceADUContent, 
                "TargetADU": targetADUContent,
                "SourceADU_tokens": self.myTokenizer(sourceADUContent, lowercase= True, removePunctuationMarks= False),
                "TargetADU_tokens": self.myTokenizer(targetADUContent, lowercase= True, removePunctuationMarks= False),
                "ArticleId": (self.validationSet).data[learningInstanceIndex]["ArticleId"]
                })
            
            (validationSetTranslations.target).append((self.validationSet).target[learningInstanceIndex])
            
        
        print "\n Translating test set ..."
        
        testSetTranslations= Bunch()
        testSetTranslations.data= []
        testSetTranslations.target= []
        testSetTranslations.target_names= ["None", "Support"]
        
        # training set
        for learningInstanceIndex in xrange(len((self.testSet).target)):
            
            sourceADUContent= str(tb((self.testSet).data[learningInstanceIndex]["SourceADU"]).translate(from_lang= sourceLanguage, to= targetLanguage))
            targetADUContent= str(tb((self.testSet).data[learningInstanceIndex]["TargetADU"]).translate(from_lang= sourceLanguage, to= targetLanguage))
            
            (testSetTranslations.data).append({
                "SourceADU": sourceADUContent, 
                "TargetADU": targetADUContent,
                "SourceADU_tokens": self.myTokenizer(sourceADUContent, lowercase= True, removePunctuationMarks= False),
                "TargetADU_tokens": self.myTokenizer(targetADUContent, lowercase= True, removePunctuationMarks= False),
                "ArticleId": (self.testSet).data[learningInstanceIndex]["ArticleId"]
                })
            
            (testSetTranslations.data).target.append((self.testSet).target[learningInstanceIndex])
            
        
        
        
        translationsFile= codecs.open(self.corpusPath + "/" + self.corpusFilename + "_translatedTo_" + targetLanguage + ".pkl", mode= "w" , encoding= "utf-8")
        pickle.dump({"trainingSet": trainingSetTranslations, "validationSet": validationSetTranslations, "testSet": testSetTranslations}, translationsFile)
        translationsFile.close()
        
        print "[Done] Translations"
        
    

