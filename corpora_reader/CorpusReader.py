#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CorpusReader: generate dataset for Argumentative Relation identification from different corpora annotated with argumentative relations
"""

from abc import ABCMeta, abstractmethod
import os
import codecs
import math
import random

from nltk.tokenize import sent_tokenize
import pandas as pd

class CorpusReader:
    __metaclass__= ABCMeta
    
    def __init__(self, corpusPath= os.path.abspath("../data/corpora/en/essays/"), generatedDatasetPath= os.path.abspath("../data/generatedDatasets/en/essays/")):
        self.corpusPath= corpusPath
        self.generatedDatasetPath= generatedDatasetPath
        
        # Dataset Partitions
        # Each element here will contain the list of ArticleIds that are assigned to the corresponding partition
        self.trainingSet= None
        self.validationSet= None
        self.testSet= None
    
    @abstractmethod
    def getADUContentWithContext(self, articleId, targetADUId, sentenceIndexes, dictADUs):
        """
        Description: given an adu id (targetADUId) for a given article (articleId) return the ADU content plus the text preceding the ADU until the begin of the sentence or the ADU that precedes targetADUId
        """
    
    @abstractmethod
    def getAnnotationForArticle(self, articleFilename):
        """
        """
    
    @abstractmethod
    def getRelationsForArticle(self, articleId, annotationsDict, includeStanceAnnotations= True, includeContext= False):
        """
        """
    
    def determineSentenceIndexes(self, articleFilename, textLanguage= "english"):
        
        sentenceBeginIndexes= []
        
        articleFiletext= codecs.open(filename= self.corpusPath + "/" + articleFilename + ".txt", mode= "r", encoding= "utf-8")
        
        articleCompleteContent= articleFiletext.read()
        
        splittedContentByEndLine= articleCompleteContent.split("\n")
        
        sentences= sent_tokenize(text= "\n".join([textSpan for textSpan in splittedContentByEndLine[2:]]), language= textLanguage)
        
        for sentence in sentences:
            sentenceBeginIndexes.append(articleCompleteContent.find(sentence))
        
        articleFiletext.close()
        
        return sentenceBeginIndexes
    
    # path= path to corpus
    # fileExtension= files extension (e.g. ".txt")
    def getAllFilenamesFromDir(self, path, fileExtension):
        
        filenames= []
        
        for textFile in os.listdir(path):
            if textFile.endswith(fileExtension):
                currentFileName= textFile.split(fileExtension)[0]
                filenames.append(currentFileName)
        return filenames
    
    # Description: for all annotations files contained in "self.corpusPath", extract relations between ADUs and create corresponding filename.csv file containing the argumentative relations dataset
    # Inputs:
    #    - filename: name of the .csv file that will contain the content of the dataset
    #    - includeContext: If True, for each ADU, we add the text between the ADU and the ADU occurring immediately before or the begin of sentence (denoted as "context" here). 
    #                      If False, only the ADUs exact token level contents are stored.
    def toCSV(self, filename, annotationsFileExtension, includeStanceAnnotations= True, includeContext= False):
        
        relationsDataframe= pd.DataFrame(columns= ["ArticleId", "SourceADU", "TargetADU", "RelationType"])
        
        for currentFilename in self.getAllFilenamesFromDir(self.corpusPath, annotationsFileExtension):
            
            essayAnnotations= self.getAnnotationForArticle(currentFilename)
            
            relationsInArticle= []
            
            for relation in self.getRelationsForArticle(essayAnnotations["articleId"], essayAnnotations, includeStanceAnnotations= includeStanceAnnotations, includeContext= includeContext):
                relationsInArticle.append([relation[0],relation[1], relation[2], relation[3]])
            
            relationsDataframe= pd.concat([relationsDataframe, pd.DataFrame(relationsInArticle, columns= ["ArticleId", "SourceADU", "TargetADU", "RelationType"])])
        
        relationsDataframe.to_csv(path_or_buf= self.generatedDatasetPath + "/" + filename + ".csv", sep= "\t", mode= "w", encoding= "utf-8", index= False)
    
    def determineDatasetSplit(self, filename, distributionRatioTolerance= 0.5, trainingSetSize= 0.8, restrictedListOfArticles= None):
        
        # load csv file to pandas
        corpus= pd.read_csv(self.generatedDatasetPath + "/" + filename + ".csv", sep= "\t")
        
        if restrictedListOfArticles is not None:
            corpus= corpus[corpus["ArticleId"].isin(restrictedListOfArticles)]
        
        supportRelations= corpus[corpus["RelationType"] == "supports"].groupby("ArticleId")["RelationType"].count()
        noneRelations= corpus[corpus["RelationType"] == "none"].groupby("ArticleId")["RelationType"].count()
        #attackRelations= corpus[corpus["RelationType"] == "attacks"].groupby("ArticleId")["RelationType"].count()
        
        totalSupportRelations= corpus[corpus["RelationType"] == "supports"].count()["RelationType"]
        totalNoneRelations= corpus[corpus["RelationType"] == "none"].count()["RelationType"]
        #totalAttackRelations= corpus[corpus["RelationType"] == "attacks"].count()["RelationType"]
        
        datasetDistRatio= float(totalSupportRelations) / float(totalNoneRelations)
        
        essayIds= corpus["ArticleId"].unique()
        
        numberOfTrainingExamples= int(math.ceil(len(essayIds) * trainingSetSize))
        
        foundSplit= False
        
        testSetEssays= []
        trainingSetEssays= []
        
        # try to split training and test set according to the desired dimensions of the training set ("trainingSetSize" parameter)
        # accept split if distributions of relations is okay
        # reject (and try a new one) otherwise
        while not foundSplit:
            # randomly suffle essay ids and assign the first x examples to the test set (remaining are training set)
            random.shuffle(essayIds)
            testSetEssays= essayIds[0:len(essayIds) - numberOfTrainingExamples]
            
            # test set distribution
            testSetDistRatio= 0.0
            
            totalSupportRelationsTestSet= 0
            totalNoneRelationsTestSet= 0
            
            for currentEssayId in testSetEssays:
                totalSupportRelationsTestSet=  totalSupportRelationsTestSet + supportRelations[currentEssayId]
                totalNoneRelationsTestSet= totalNoneRelationsTestSet + noneRelations[currentEssayId]
            
            if (totalSupportRelationsTestSet > totalNoneRelationsTestSet):
                testSetDistRatio= float(totalNoneRelationsTestSet) / float(totalSupportRelationsTestSet)
            else:
                testSetDistRatio= float(totalSupportRelationsTestSet) / float(totalNoneRelationsTestSet)
            
            
            if abs(testSetDistRatio - datasetDistRatio) < distributionRatioTolerance:
                trainingSetEssays= [essayId for essayId in essayIds if essayId not in testSetEssays]
                
                # training set distribution
                trainSetDistRatio= 0.0
                
                totalSupportRelationsTrainSet= 0
                totalNoneRelationsTrainSet= 0
                
                for currentEssayId in trainingSetEssays:
                    totalSupportRelationsTrainSet=  totalSupportRelationsTrainSet + supportRelations[currentEssayId]
                    totalNoneRelationsTrainSet= totalNoneRelationsTrainSet + noneRelations[currentEssayId]
                
                if (totalSupportRelationsTrainSet > totalNoneRelationsTrainSet):
                    trainSetDistRatio= float(totalNoneRelationsTrainSet) / float(totalSupportRelationsTrainSet)
                else:
                    trainSetDistRatio= float(totalSupportRelationsTrainSet) / float(totalNoneRelationsTrainSet)
                
                if abs(trainSetDistRatio - datasetDistRatio) < distributionRatioTolerance:
                    foundSplit= True
        
        
        return (trainingSetEssays, testSetEssays)
    
    def addDatasetPartitions(self, filename, trainTestSetSplit= None, trainValidationSplit= None):
        
        if (trainTestSetSplit is not None) and (trainValidationSplit is not None):
            tempTrainSet, self.testSet= self.determineDatasetSplit(filename, trainingSetSize= trainTestSetSplit, restrictedListOfArticles= None)
            
            self.trainingSet, self.validationSet= self.determineDatasetSplit(filename, trainingSetSize= trainValidationSplit, restrictedListOfArticles= tempTrainSet)
        
        # load csv file to pandas
        corpus= pd.read_csv(self.generatedDatasetPath + "/" + filename + ".csv", sep= "\t")
        
        partitionColumn= [self.determinePartition(row.ArticleId, self.trainingSet, self.validationSet, self.testSet) for index, row in corpus.iterrows()]
        
        corpus["Partition"]= partitionColumn
        
        corpus.to_csv(path_or_buf= self.generatedDatasetPath + "/" + filename + ".csv", sep= "\t", mode= "w", encoding= "utf-8", index= False)
    
    def determinePartition(self, articleId, trainingSetArticleIds, validationSetArticleIds, testSetArticleIds):
        
        if articleId in trainingSetArticleIds:
            return "TRAIN"
        elif articleId in validationSetArticleIds:
            return "VALIDATION"
        elif articleId in testSetArticleIds:
            return "TEST"
        else:
            raise Exception("Article with id= " + str(articleId) + " not found in any of the partitions!")
        
    
    def corpusStats(self, filename):
        
        # load csv file
        corpus= pd.read_csv(self.generatedDatasetPath + "/" + filename + ".csv", sep= "\t")
        
        #print corpus.describe()
        
        print "\nRelations stats:"
        #print corpus["RelationType"].describe()
        
        print "# nones's= " + str(len([elem for elem in corpus["RelationType"] if elem == "none"]))
        print "# support's= " + str(len([elem for elem in corpus["RelationType"] if elem == "supports"]))
        print "# attack's= " + str(len([elem for elem in corpus["RelationType"] if elem == "attacks"]))
        
        print "Group by essays"
        #print corpus.groupby(["ArticleId"]).describe()
        
        aggregation= {
                "count": "count",
                "sum": "sum",
                "mean": "mean",
                "median": "median",
                "std": "std",
                "min": "min",
                "max": "max",
            }
        
        print (corpus.groupby("ArticleId", as_index= False)["RelationType"].count())["RelationType"].agg(aggregation)
    
    
    def createDatasetForArgumentativeRelationIdentification(self, annotationsFileExtension= ".json", csvFilename= "ArgEssaysCorpus_en", csvFilenameWithContext= "ArgEssaysCorpus_context_en", includeStanceAnnotations= True):
        # create dataset
        self.toCSV(csvFilename, annotationsFileExtension, includeStanceAnnotations= includeStanceAnnotations, includeContext= False)
        
        print "\nDataset stats for generated dataset entitled '" + str(csvFilename) + "':"
        self.corpusStats(csvFilename)
        
        self.addDatasetPartitions(csvFilename, trainTestSetSplit= 0.8, trainValidationSplit= 0.8)
        
        # create dataset with context
        # context= for a given ADU, attach the text until the end of the previous ADU or until the end of sentence
        
        self.toCSV(csvFilenameWithContext, annotationsFileExtension, includeStanceAnnotations= includeStanceAnnotations, includeContext= True)
        
        print "\nDataset stats for generated dataset entitled '" + str(csvFilenameWithContext) + "':"
        self.corpusStats(csvFilenameWithContext)
        
        # Parameters are None, meaning that they will use the partitions already determined and stored in self.trainingSet, self.testSet and self.validationSet
        self.addDatasetPartitions(csvFilenameWithContext, trainTestSetSplit= None, trainValidationSplit= None)
        
    
    