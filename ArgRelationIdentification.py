#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Argumentative Relation Identification
"""


import os
import codecs
import sys

import numpy as np
import pandas as pd
import pickle
import random

from keras import backend as K
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


from sklearn.utils import shuffle
from sklearn.datasets.base import Bunch
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier

import DatasetLoader
import SentenceEncoding
import NeuralNetworkArchitectures

# Paths
currentPath= os.path.dirname(os.path.abspath(__file__))
logsPath= os.path.abspath("data/logs")
modelsPath= os.path.abspath("data/models")
errorAnalysisPath= os.path.abspath("data/errorAnalysis")


class ArgumentativeRelationClassifier:
    
    def __init__(self, datasetPath= os.path.abspath("data/generatedDatasets/en/essays/"), datasetFilename= "ArgEssaysCorpus_en", language= "en"):
        print "\n\n+++ Argumentative Relation Classifier +++\n\n"
        
        self.datasetPath= datasetPath
        
        self.datasetFilename= datasetFilename
        
        self.language= language
        
        self.randomSeed= 12345
        
        self.datasetLoader= DatasetLoader.DatasetLoader(corpusPath= self.datasetPath, corpusFilename= self.datasetFilename)
        
        if "translatedTo" in self.datasetFilename:
            translationsFile= codecs.open(self.datasetPath + "/" + self.datasetFilename + ".pkl", mode= "r")
            translationsDict= pickle.load(translationsFile)
            self.trainingSet= translationsDict["trainingSet"]
            self.validationSet= translationsDict["validationSet"]
            self.testSet= translationsDict["testSet"]
        else:
            (self.datasetLoader).loadDataFromCSV()
            
            self.trainingSet= (self.datasetLoader).trainingSet
            self.validationSet= (self.datasetLoader).validationSet
            self.testSet= (self.datasetLoader).testSet
        
        (self.trainingSet).data, (self.trainingSet).target = shuffle((self.trainingSet).data, (self.trainingSet).target, random_state= self.randomSeed)
        (self.validationSet).data, (self.validationSet).target = shuffle((self.validationSet).data, (self.validationSet).target, random_state= self.randomSeed)
        (self.testSet).data, (self.testSet).target = shuffle((self.testSet).data, (self.testSet).target, random_state= self.randomSeed)
        
        
        self.completeDataset= Bunch()
        self.completeDataset.data= np.append(np.append((self.trainingSet).data, (self.validationSet).data), (self.testSet).data)
        self.completeDataset.target= np.append(np.append((self.trainingSet).target, (self.validationSet).target), (self.testSet).target)
        self.completeDataset.target_names= (self.trainingSet).target_names
        
        #(self.datasetLoader).getCrossValidationSplits(self.completeDataset, nSplits= 5)
        
        
        print "\n Dataset Loaded Successfuly ...\n"
        
    
    # Transforms the datasets received as input into learning instances in a format appropriated for processing in the next steps (words to embeddings, padding, ...)
    # If run for the first time it creates everything from scratch. Otherwise, it loads several files (for different parts of the preprocessing) already previously created.
    def preProcessing(self, completeDataset= None, trainingSet= None, validationSet= None, testSet= None, sentenceEncodingType= "conventional", experimentName= "experiment"):
        
        # Sentence encoding procedure 
        self.sentenceEncoding_= SentenceEncoding.SentenceEncoding(
            dataset= self.completeDataset,
            language= self.language,
            name= sentenceEncodingType,
            experimentName= experimentName,
            datasetPath= self.datasetPath,
            datasetFilename= self.datasetFilename
            )
        
        if completeDataset is None:
            # Assumes that we provide the splits for training, validation and test set
            # Typically used to prepare data for in-language runs
            
            # training set
            trainingSet.data= self.sentenceEncoding_.transform(trainingSet.data)
            
            # test set
            testSet.data= self.sentenceEncoding_.transform(testSet.data)
            
            # validation set
            if validationSet is not None:
                validationSet.data= self.sentenceEncoding_.transform(validationSet.data)
                return (trainingSet, validationSet, testSet)
            else:
                return (trainingSet, testSet)
        else:
            # ignores the fields trainingSet, validationSet and testSet.
            # Typically used in the 2nd step of Direct Transfer procedures (to preprocess the data of the target language)
            completeDataset.data= self.sentenceEncoding_.transform(completeDataset.data)
            return completeDataset
    
    # Returns the deep learning architecture to be used in the experiments
    def buildNeuralNetworkArchitecture(self, neuralNetArchitecture= "SumOfEmbeddings_Concatenation_1Layer", numberOfClasses= 2, loadModelFilename= None, loadEmbeddingsMatrixFilename= None, randomSeedValue= 0):
        
        if loadModelFilename is None:
            # constructs neural net architecture from scratch 
            # All weights, except the Embeddings Layer which is initialized based on a pre-trained word embeddings model, are initialized with the default values (as defined in keras)
            model= NeuralNetworkArchitectures.neuralNetwork(
                architectureName= neuralNetArchitecture, 
                numberOfClasses= numberOfClasses,
                embeddingsMatrixPath= self.datasetPath + "/WordEmbeddings/",
                embeddingsMatrixFilename= self.datasetFilename + "_EmbeddingsMatrix",
                randomSeedValue= randomSeedValue
                )
        else:
            # Create neural network architecture and initialize the weights based on saved model at: modelsPath + "/" + str(loadModelFilename)
            
            # create a "clone" model that will be loaded with all the weights stored from by an existing model at: modelsPath + "/" + str(loadModelFilename)
            sourceModel= NeuralNetworkArchitectures.neuralNetwork(
                architectureName= neuralNetArchitecture, 
                numberOfClasses= numberOfClasses,
                embeddingsMatrixPath= self.datasetPath + "/WordEmbeddings/",
                embeddingsMatrixFilename= loadEmbeddingsMatrixFilename,
                randomSeedValue= randomSeedValue
                )
            
            sourceModel.load_weights(filepath= modelsPath + "/" + str(loadModelFilename))
            
            # we want to load all the weights except the embeddings layer which will be initialized with new weights (for the target language instead of the source language)
            model= NeuralNetworkArchitectures.neuralNetwork(
                architectureName= neuralNetArchitecture, 
                numberOfClasses= numberOfClasses,
                embeddingsMatrixPath= self.datasetPath + "/WordEmbeddings/",
                embeddingsMatrixFilename= self.datasetFilename + "_EmbeddingsMatrix",
                randomSeedValue= randomSeedValue
                )
            
            # update weights of "model" with the weights in "sourceModel", except the embeddings layer which are already updated for the target language
            for layer in sourceModel.layers:
                if not ( (layer.name.startswith("Premise_EmbedLayer")) or (layer.name.startswith("Claim_EmbedLayer"))):
                    model.get_layer(layer.name).set_weights(layer.get_weights())
            
        
        return model
        
    
    # All the logic for training the models lies here
    def learn(self, experimentName= "experiment", modelFilename= "deepModel", numberOfClasses= 2, sentenceEncodingType= "conventional", neuralNetArchitectureName= "SumOfEmbeddings_Concatenation_1Layer", loadModelFilename= None, loadEmbeddingsMatrixFilename= None, balanceDataset= False, classWeight= False, randomSeedValue= 0):
        
        print "\n ... Learning ...\n"
        
        # Dealing with unbalance datasets
        #    - Random Undersampling
        # check whether we want to balance the training set
        if balanceDataset:
            self.trainingSet= (self.datasetLoader).randomUndersamplingForBinaryClassification((self.trainingSet), self.randomSeed)
        
        # PreProcessing step
        self.trainingSet, self.validationSet, self.testSet= self.preProcessing(
            trainingSet= self.trainingSet, 
            validationSet= self.validationSet, 
            testSet= self.testSet, 
            sentenceEncodingType= sentenceEncodingType,
            experimentName= experimentName
            )
        
        # Construct Neural Network Architecture
        self.model= self.buildNeuralNetworkArchitecture(
            numberOfClasses= numberOfClasses,
            neuralNetArchitecture= neuralNetArchitectureName,
            loadModelFilename= loadModelFilename,
            loadEmbeddingsMatrixFilename= loadEmbeddingsMatrixFilename,
            randomSeedValue= randomSeedValue
            )
        
        # Callbacks to be added during training
        runCallbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=0), 
            ModelCheckpoint(modelsPath + "/" + modelFilename + ".h5", monitor='val_loss', save_best_only=True, verbose=0, save_weights_only= True),
            #TensorBoard(log_dir= logsPath)
            ]
        
        
        if classWeight:
            # Dealing with unbalance datasets
            #    - Cost sensitive learning
            self.model.fit(
                (self.trainingSet).data, 
                to_categorical(y= (self.trainingSet).target, num_classes= numberOfClasses),
                epochs= 50, 
                batch_size= 32, 
                class_weight= computeClassWeights((self.trainingSet).target),
                validation_data= (
                    (self.validationSet).data, 
                    to_categorical(y= (self.validationSet).target, num_classes= numberOfClasses)),
                callbacks= runCallbacks,
                verbose= 2
                )
        else:
            self.model.fit(
                (self.trainingSet).data, 
                to_categorical(y= (self.trainingSet).target, num_classes= numberOfClasses),
                epochs= 50,
                batch_size= 32,
                validation_data= (
                    (self.validationSet).data,
                    to_categorical(y= (self.validationSet).target, num_classes= numberOfClasses)),
                callbacks= runCallbacks,
                verbose= 2
                )
        
        
        # Load parameters for best model seen during training
        (self.model).load_weights(filepath= modelsPath + "/" + modelFilename + ".h5")
        
        print " \n\n ##### Training Set - Evaluation Metrics #####\n\n"
        print "\n Obtaining predictions ...\n"
        
        #TODO: Deprecated in the current version. Left because it might be useful in the future
        predictionsDict= {"neuralNetArchitectureName": neuralNetArchitectureName, "numberOfClasses": numberOfClasses, "modelFilename": modelFilename + ".h5"}
        
        trainingSetPredictions= self.getPredictions(self.model, self.trainingSet, predictionsDict)
        self.evaluationMetrics((self.trainingSet).target, trainingSetPredictions, (self.trainingSet).target_names)
        
        print " \n\n ##### Validation Set - Evaluation Metrics #####\n\n"
        print "\n Obtaining predictions ...\n"
        
        validationSetPredictions= self.getPredictions(self.model, self.validationSet, predictionsDict)
        self.evaluationMetrics((self.validationSet).target, validationSetPredictions, (self.validationSet).target_names)
        
        print " \n\n ##### Test Set - Evaluation Metrics #####\n\n"
        print "\n Obtaining predictions ...\n"
        
        testSetPredictions= self.getPredictions(self.model, self.testSet, predictionsDict)
        self.evaluationMetrics((self.testSet).target, testSetPredictions, (self.testSet).target_names)
        
        print "\nThe end!"
    
    
    def getPredictions(self, model, testSet, parameters):
        
        """
        print "\n\n Loading model ..."
        
        # Load back the pickled model
        ##model= load_model(filepath= modelsPath + "/" + str(modelFilename))
        model= self.neuralNet(
            architectureName= parameters["neuralNetArchitectureName"], ##neuralNetArchitecture, 
            numberOfClasses= parameters["numberOfClasses"], ##numberOfClasses, 
            ##embeddingsVectorDimension= (self.sentenceEncoding_).wordEmbeddingsDimensions, 
            ##vocabularySize= (self.sentenceEncoding_).vocabularySize,
            embeddingsMatrixFilename= self.datasetFilename + "_EmbeddingsMatrix"
            )
        
        model.load_weights(filepath= modelsPath + "/" + parameters["modelFilename"]) ##str(loadModelFilename))
        
        print "\n\n Model loaded successfully"
        """
        
        print "\n\n Obtaining Prediction ..."
        
        predictions= model.predict(testSet.data)
        
        predictionsTargetLabels= np.argmax(predictions, axis= 1)
        ###predictionsTargetLabels= [1 if elem[0] > 0.5 else 0 for elem in predictions]
        
        print "\n\n Predictions: done."
        
        return predictionsTargetLabels
        
    
    
    def evaluationMetrics(self, y, predicted, targetNames, title= ""):
        
        print "\n" + title + "\n"
        
        # comparing prediction with the known categories in order to determine accuracy
        datasetAccuracy= np.mean(predicted == y)
        
        print "Accuracy= " + str(datasetAccuracy)
        
        # +++ Detailed Performance Analysis +++
        
        # Confusion Matrix table
        print "\nConfusion Matrix:"
        
        
        print metrics.confusion_matrix(y, predicted)
        
        print ""
        
        print(metrics.classification_report(y, predicted, target_names= targetNames))
    
    
    # Feature-based approach for in-language Argumentative Relation Identification with Cross-Validation
    # Baseline approach
    # Currently only considers Bag-Of-Words with a simple classifier in the end (logistic regression, svm, ...)
    def singleLanguageLearn_FeatureBasedApproach(self, completeDataset, experimentName= "experiment", modelFilename= "myModel", numberOfClasses= 2, numberOfSplits= 5, balanceDataset= False, classWeight= False):
        
        datasetPartitionsFile= codecs.open(str(self.datasetPath) + "/" + "FoldsPartition" + "/" + str(self.datasetFilename) + "_DatasetPartition_" + str(numberOfSplits) + "Folds.pkl", mode= "r")
        
        cvFolds= pickle.load(datasetPartitionsFile)
        
        datasetPartitionsFile.close()
        
        predicted= []
        goldenStandardCurrentFold= []
        
        confusionMatrixesListTrainingSet= []
        confusionMatrixesListValidationSet= []
        confusionMatrixesListTestSet= []
        
        iteration= 0
        
        for trainingSetCurrentFold, validationSetCurrentFold, testSetCurrentFold in cvFolds:
            
            # check whether we want to balance the training set
            if balanceDataset:
                trainingSetCurrentFold= (self.datasetLoader).randomUndersamplingForBinaryClassification(trainingSetCurrentFold, self.randomSeed)
            
            # Pre-Processing data
            
            # uses tokenization performed while loading the dataset (e.g. employing external model for tokenization)
            #trainingSetCurrentFold.data= np.asarray([" ".join(learningInstance["SourceADU_tokens"]) + " DELIM " + " ".join(learningInstance["TargetADU_tokens"]) for learningInstance in trainingSetCurrentFold.data])
            #validationSetCurrentFold.data= np.asarray([" ".join(learningInstance["SourceADU_tokens"]) + " DELIM " + " ".join(learningInstance["TargetADU_tokens"]) for learningInstance in validationSetCurrentFold.data])
            #testSetCurrentFold.data= np.asarray([" ".join(learningInstance["SourceADU_tokens"]) + " DELIM " + " ".join(learningInstance["TargetADU_tokens"]) for learningInstance in testSetCurrentFold.data])
            
            # uses Keras preprocessing utilities
            from keras.preprocessing.text import text_to_word_sequence
            trainingSetCurrentFold.data= np.asarray([" ".join(text_to_word_sequence(learningInstance["SourceADU"])) + " DELIM " + " ".join(text_to_word_sequence(learningInstance["TargetADU"])) for learningInstance in trainingSetCurrentFold.data])
            validationSetCurrentFold.data= np.asarray([" ".join(text_to_word_sequence(learningInstance["SourceADU"])) + " DELIM " + " ".join(text_to_word_sequence(learningInstance["TargetADU"])) for learningInstance in validationSetCurrentFold.data])
            testSetCurrentFold.data= np.asarray([" ".join(text_to_word_sequence(learningInstance["SourceADU"])) + " DELIM " + " ".join(text_to_word_sequence(learningInstance["TargetADU"])) for learningInstance in testSetCurrentFold.data])
            
            # create pipeline
            features= FeatureUnion([
                          ('ngram', CountVectorizer(analyzer="word", ngram_range=(1, 3),  decode_error= 'ignore'))
                          ],
                         )
            
            # Classifier
            if classWeight:
                model= LogisticRegression(class_weight= 'balanced')
            else:
                model= LogisticRegression()
            #NOTE: to get random baseline predictions, simply change "model" to DummyClassifier(strategy="uniform", random_state= iteration) #DummyClassifier(strategy="stratified", random_state=iteration)
            
            
            pipeline= Pipeline([("features", features), ("clf", model)])
            
            pipeline= pipeline.fit(trainingSetCurrentFold.data, trainingSetCurrentFold.target)
            
            
            # Test set - evaluation metrics
            print "\n--- Iteration #" + str(iteration)
            
            
            print "Training Set - Evaluation Metrics"
            currentPredictions= pipeline.predict(trainingSetCurrentFold.data)
            self.evaluationMetrics(trainingSetCurrentFold.target, currentPredictions, trainingSetCurrentFold.target_names)
            
            confusionMatrixesListTrainingSet.append(metrics.confusion_matrix(trainingSetCurrentFold.target, currentPredictions))
            
            
            print "Validation Set - Evaluation Metrics"
            currentPredictions= pipeline.predict(validationSetCurrentFold.data)
            #currentPredictions= np.argmax(currentPredictions, axis= 1)
            self.evaluationMetrics(validationSetCurrentFold.target, currentPredictions, trainingSetCurrentFold.target_names)
            
            ###confusionMatrixesListValidationSet.append(metrics.confusion_matrix(completeDataset.target[validationSetIdx], currentPredictions))
            confusionMatrixesListValidationSet.append(metrics.confusion_matrix(validationSetCurrentFold.target, currentPredictions))
            
            print "Test Set - Evaluation Metrics"
            currentPredictions= pipeline.predict(testSetCurrentFold.data)
            #currentPredictions= np.argmax(currentPredictions, axis= 1)
            
            self.evaluationMetrics(testSetCurrentFold.target, currentPredictions, testSetCurrentFold.target_names)
            
            confusionMatrixesListTestSet.append(metrics.confusion_matrix(testSetCurrentFold.target, currentPredictions))
            
            print "confusion matrix:"
            print metrics.confusion_matrix(testSetCurrentFold.target, currentPredictions)
            
            #predicted[testIdx]= currentPredictions
            predicted.append(currentPredictions)
            goldenStandardCurrentFold.append(testSetCurrentFold.target)
            
            
            iteration = iteration + 1
            
        
        # compute final confusion matrixes -> sum of the confusion matrixes extracted from each iteration
        finalConfusionMatrixTrainingSet= np.sum(confusionMatrixesListTrainingSet, axis=0)
        
        finalConfusionMatrixValidationSet= np.sum(confusionMatrixesListValidationSet, axis=0)
        
        finalConfusionMatrixTestSet= np.sum(confusionMatrixesListTestSet, axis=0)
        
        print "\n\n **** Final Confusion Matrix - Test Set ****\n"
        print finalConfusionMatrixTestSet
        
        
        # store relevant test set data in pickle file
        testSetInfoFile= codecs.open(logsPath + "/" + modelFilename + "_featureBased_testSetPredictions.pkl", mode= "w", encoding= "utf-8")
        
        testSetInfoDict= {
            "predictions": np.asarray(predicted),
            "targetGoldStandard": np.asarray(completeDataset.target),
            "dataInstances": np.asarray([np.asarray([elem for elem in (completeDataset.data)[0]]), np.asarray([elem for elem in (completeDataset.data)[1]])]),
            "confusionMatrixes": np.asarray(confusionMatrixesListTestSet)
            }
        
        pickle.dump(testSetInfoDict, testSetInfoFile)
        
        testSetInfoFile.close()
        
        return {"trainingSet": {"finalConfusionMatrix": finalConfusionMatrixTrainingSet}, "validationSet": {"finalConfusionMatrix": finalConfusionMatrixValidationSet}, "testSet": {"predictions": predicted, "finalConfusionMatrix": finalConfusionMatrixTestSet}}
    
    # In-Language Argumentative Relation Identification using the Neural Network Architectures also used in cross-language experiments
    # Differs from learn() by running the experiments in a cross-validation setting (learn() runs it once for a given training, validation and test set partition).
    def singleLanguageLearn(self, completeDataset, experimentName= "experiment", modelFilename= "myModel", numberOfClasses= 2, numberOfSplits= 5, sentenceEncodingType= "conventional", neuralNetArchitectureName= "SumOfEmbeddings_Concatenation_1Layer", balanceDataset= False, classWeight= False):
        
        datasetPartitionsFile= codecs.open(str(self.datasetPath) + "/" + "FoldsPartition" + "/" + str(self.datasetFilename) + "_DatasetPartition_" + str(numberOfSplits) + "Folds.pkl", mode= "r")
        
        cvFolds= pickle.load(datasetPartitionsFile)
        
        datasetPartitionsFile.close()
        
        predicted= []
        goldenStandardCurrentFold= []
        
        confusionMatrixesListTrainingSet= []
        confusionMatrixesListValidationSet= []
        confusionMatrixesListTestSet= []
        
        iteration= 0
        
        testIdxByIteration= []
        
        for trainingSetCurrentFold, validationSetCurrentFold, testSetCurrentFold in cvFolds:
            
            # Dealing with unbalance datasets
            #    - Random Undersampling
            # check whether we want to balance the training set
            if balanceDataset:
                trainingSetCurrentFold= (self.datasetLoader).randomUndersamplingForBinaryClassification(trainingSetCurrentFold, self.randomSeed)
                
            
            # PreProcessing step
            trainingSetCurrentFold, validationSetCurrentFold, testSetCurrentFold = self.preProcessing(
                trainingSet= trainingSetCurrentFold, 
                validationSet= validationSetCurrentFold, 
                testSet= testSetCurrentFold, 
                sentenceEncodingType= sentenceEncodingType,
                experimentName= experimentName)
            
            # Construct Neural Network Architecture
            pipeline= self.buildNeuralNetworkArchitecture(
                numberOfClasses= numberOfClasses, 
                neuralNetArchitecture= neuralNetArchitectureName,
                loadModelFilename= None
                )
            
            # Callbacks to be added during training
            runCallbacks = [
                EarlyStopping(monitor='val_loss', patience=15, verbose=0), 
                ModelCheckpoint(modelsPath + "/" + modelFilename + ".h5", monitor='val_loss', save_best_only=True, verbose=0, save_weights_only= True),
                #TensorBoard(log_dir= logsPath)
            ]
            
            if classWeight:
                # Dealing with unbalance datasets
                #    - Cost sensitive learning
                pipeline.fit(
                    trainingSetCurrentFold.data, 
                    to_categorical(y= trainingSetCurrentFold.target, num_classes= numberOfClasses),
                    epochs= 50, 
                    batch_size= 32,
                    class_weight= computeClassWeights(trainingSetCurrentFold.target),
                    validation_data= (
                        validationSetCurrentFold.data,
                        to_categorical(y= validationSetCurrentFold.target, num_classes= numberOfClasses)
                        ),
                    callbacks= runCallbacks,
                    verbose= 2
                )
            else:
                pipeline.fit(
                    trainingSetCurrentFold.data, 
                    to_categorical(y= trainingSetCurrentFold.target, num_classes= numberOfClasses),
                    epochs= 50, 
                    batch_size= 32, 
                    #validation_split= 0.2,
                    validation_data= (
                        validationSetCurrentFold.data,
                        to_categorical(y= validationSetCurrentFold.target, num_classes= numberOfClasses)
                        ),
                    callbacks= runCallbacks,
                    verbose= 2
                )
            
            
            # Load parameters for best model seen during training
            pipeline.load_weights(filepath= modelsPath + "/" + modelFilename + ".h5")
            
            # Test set - evaluation metrics
            print "\n--- Iteration #" + str(iteration)
            
            
            print "Training Set - Evaluation Metrics"
            currentPredictions= pipeline.predict(trainingSetCurrentFold.data)
            currentPredictions= np.argmax(currentPredictions, axis= 1)
            self.evaluationMetrics(trainingSetCurrentFold.target, currentPredictions, trainingSetCurrentFold.target_names)
            
            confusionMatrixesListTrainingSet.append(metrics.confusion_matrix(trainingSetCurrentFold.target, currentPredictions))
            
            
            print "Validation Set - Evaluation Metrics"
            currentPredictions= pipeline.predict(validationSetCurrentFold.data)
            currentPredictions= np.argmax(currentPredictions, axis= 1)
            
            self.evaluationMetrics(validationSetCurrentFold.target, currentPredictions, trainingSetCurrentFold.target_names)
            
            confusionMatrixesListValidationSet.append(metrics.confusion_matrix(validationSetCurrentFold.target, currentPredictions))
            
            print "Test Set - Evaluation Metrics"
            currentPredictions= pipeline.predict(testSetCurrentFold.data)
            currentPredictions= np.argmax(currentPredictions, axis= 1)
            
            self.evaluationMetrics(testSetCurrentFold.target, currentPredictions, testSetCurrentFold.target_names)
            
            confusionMatrixesListTestSet.append(metrics.confusion_matrix(testSetCurrentFold.target, currentPredictions))
            
            print "confusion matrix:"
            print metrics.confusion_matrix(testSetCurrentFold.target, currentPredictions)
            
            #predicted[testIdx]= currentPredictions
            predicted.append(currentPredictions)
            goldenStandardCurrentFold.append(testSetCurrentFold.target)
            
            iteration = iteration + 1
        
        
        
        # compute final confusion matrixes -> sum of the confusion matrixes extracted from each iteration
        finalConfusionMatrixTrainingSet= np.sum(confusionMatrixesListTrainingSet, axis=0)
        
        finalConfusionMatrixValidationSet= np.sum(confusionMatrixesListValidationSet, axis=0)
        
        print "\n\n **** Final Confusion Matrix - Test Set ****\n"
        finalConfusionMatrixTestSet= np.sum(confusionMatrixesListTestSet, axis=0)
        
        print finalConfusionMatrixTestSet
        
        
        # store relevant test set data in pickle file
        testSetInfoFile= codecs.open(logsPath + "/" + modelFilename + "_testSetPredictions.pkl", mode= "w", encoding= "utf-8")
        
        testSetInfoDict= {
            "predictions": np.asarray(predicted),
            "targetGoldStandard": np.asarray(completeDataset.target),
            "dataInstances": np.asarray([np.asarray([elem for elem in (completeDataset.data)[0]]), np.asarray([elem for elem in (completeDataset.data)[1]])]),
            "indexesByIteration": np.asarray(testIdxByIteration),
            "confusionMatrixes": np.asarray(confusionMatrixesListTestSet)
            }
        
        pickle.dump(testSetInfoDict, testSetInfoFile)
        
        testSetInfoFile.close()
        
        return {"trainingSet": {"finalConfusionMatrix": finalConfusionMatrixTrainingSet}, "validationSet": {"finalConfusionMatrix": finalConfusionMatrixValidationSet}, "testSet": {"predictions": predicted, "finalConfusionMatrix": finalConfusionMatrixTestSet}}
        
    
        
    
    
    
    
    
    


def computeClassWeights(y):
    return dict(enumerate(compute_class_weight('balanced',np.unique(y), y)))


def randomSeedInitialization(myRandomSeed):
    print "Running with seed: " + str(myRandomSeed)
    
    # random seeds
    np.random.seed(myRandomSeed)
    
    random.seed(myRandomSeed)
    
    tf.set_random_seed(myRandomSeed)
    

def directTransferLearning(parameters= None):
    
    sys.stdout= codecs.open(logsPath + "/" + str(parameters["name"]) + "_console.txt", mode= 'w')
    
    numberOfClasses= 2
    
    nIterations= 10
    
    bestIterationPredictionsScore= -1
    
    model1ConfMatrixByIteration= []
    
    bestIterationPredictionsScoreModel2= -1
    
    model2ConfMatrixByIteration= []
    
    # delete files that will be overwritten here in append 'a' mode in each iteration
    testSetModel2InfoFile= codecs.open(logsPath + "/" + parameters["name"] + "_model2_Predictions.pkl", mode= "w", encoding= "utf-8")
    
    testSetModel2InfoDict= {
        "predictions": [],
        "targetGoldStandard": [],
        "dataInstances": [],
        "confusionMatrixes": []
        }
    
    pickle.dump(testSetModel2InfoDict, testSetModel2InfoFile)
    
    testSetModel2InfoFile.close()
    
    # nIterations= number of runs that all the experiments will be repeated with different seeds to ensure stable and reliable results when performing experiments with (Deep) Neural Networks
    for iterId in range(nIterations):
        
        ###################
        ###   Model 1   ###
        ###################
        
        randomSeedInitialization(iterId)
        
        classifier1= ArgumentativeRelationClassifier(
            datasetPath= parameters["dataset1Path"], 
            datasetFilename= parameters["dataset1Filename"], 
            language= parameters["dataset1Language"])
        
        try:
            os.remove(modelsPath + "/" + parameters["name"] + "_model1.h5")
        except OSError:
            pass
        
        classifier1.learn(
            experimentName= parameters["name"],
            modelFilename= parameters["name"] + "_model1", 
            numberOfClasses= numberOfClasses, 
            sentenceEncodingType= parameters["sentenceEncodingType"], 
            neuralNetArchitectureName= parameters["neuralNetArchitectureName"],
            loadModelFilename= None,
            balanceDataset= parameters["balanceDataset"],
            classWeight= parameters["classWeight"],
            randomSeedValue= iterId
        )
        
        predictionsDict= {"neuralNetArchitectureName": parameters["neuralNetArchitectureName"], "numberOfClasses": numberOfClasses, "modelFilename": parameters["name"] + "_model1.h5"}
        classifier1Predictions= classifier1.getPredictions(classifier1.model, classifier1.testSet, predictionsDict)
        
        classifier1.evaluationMetrics(classifier1.testSet.target, classifier1Predictions, classifier1.testSet.target_names)
        
        model1ConfMatrixByIteration.append(metrics.confusion_matrix(classifier1.testSet.target, classifier1Predictions))
        
        # save best model seen so far
        # the "best" model will be the one used in model2 (for direct transfer)
        # "best" is defined as the model with better macro f1-score on the validation set
        classifier1Predictions= classifier1.getPredictions(classifier1.model, classifier1.validationSet, predictionsDict)
        
        currentModelScore= f1_score(y_true= classifier1.validationSet.target, y_pred= classifier1Predictions, average= "macro")
        
        print "current macro f1-score (validation set)= " + str(currentModelScore)
        
        
        if os.path.isfile(modelsPath + "/" + parameters["name"] + "_best_model1.h5"):
            
            if currentModelScore > bestIterationPredictionsScore:
                print "found best classifier at iteration " + str(iterId)
                
                classifier1.model.save_weights(filepath= modelsPath + "/" + parameters["name"] + "_best_model1.h5")
                print "model weights saved successfully!"
                
                bestIterationPredictionsScore= currentModelScore
            
        else:
            # save model weights after the first iteration
            classifier1.model.save_weights(filepath= modelsPath + "/" + parameters["name"] + "_best_model1.h5")
        
        
        ###################
        ###   Model 2   ###
        ###################
        
        
        classifier2= ArgumentativeRelationClassifier(
            datasetPath= parameters["dataset2Path"], 
            datasetFilename= parameters["dataset2Filename"], 
            language= parameters["dataset2Language"])
        
        # PreProcessing step
        completeDataset= classifier2.preProcessing(
            completeDataset= classifier2.completeDataset,
            experimentName= parameters["name"],
            sentenceEncodingType= parameters["sentenceEncodingType"]
            )
        
        # Construct Neural Network Architecture
        classifier2Model= classifier2.buildNeuralNetworkArchitecture(
            neuralNetArchitecture= parameters["neuralNetArchitectureName"], 
            numberOfClasses= numberOfClasses, 
            loadModelFilename= parameters["name"]  + "_model1.h5",
            loadEmbeddingsMatrixFilename= parameters["dataset2Filename"] + "_EmbeddingsMatrix",
            randomSeedValue= iterId
            )
        
        try:
            os.remove(modelsPath + "/" + parameters["name"] + "_model2.h5")
        except OSError:
            pass
        
        # save weights of model2
        classifier2Model.save_weights(filepath= modelsPath + "/" + parameters["name"] + "_model2.h5")
        
        """
        # retraining in the target language
        #TODO: deprecated. If you want to try this out, update accordingly
        model= classifier2.learn(
            modelFilename= dataset2Filename + "_model2.h5", 
            numberOfClasses= numberOfClasses, 
            sentenceEncodingType= sentenceEncoding, 
            neuralNetArchitectureName= neuralNetArchitecture, 
            fixedWordEmbeddingsDict= ((classifier1.sentenceEncoding_).tokenizer).word_index, 
            loadModelFilename= dataset1Filename + "_model1.h5"
            )
        """
        
        predictionsDict= {"neuralNetArchitectureName": parameters["neuralNetArchitectureName"], "numberOfClasses": numberOfClasses, "modelFilename": parameters["name"] + "_model2.h5"}
        classifier2Predictions= classifier2.getPredictions(classifier2Model, completeDataset, predictionsDict)
        
        classifier2.evaluationMetrics(completeDataset.target, classifier2Predictions, completeDataset.target_names)
        
        currentModel2Score= f1_score(y_true= completeDataset.target, y_pred= classifier2Predictions, average= "macro")
        
        print "current macro f1-score (target dataset)= " + str(currentModel2Score)
        
        model2ConfMatrixCurrentIteration= metrics.confusion_matrix(completeDataset.target, classifier2Predictions)
        model2ConfMatrixByIteration.append(model2ConfMatrixCurrentIteration)
        
        # update test set model predictions file
        # read to get content of previous iterations -> update with content from the current iteration + save new content
        testSetModel2InfoFile= codecs.open(logsPath + "/" + parameters["name"] + "_model2_Predictions.pkl", mode= "r")
        
        currentMode2InfoFileContent= pickle.load(testSetModel2InfoFile)
        
        testSetModel2InfoFile.close()
        
        testSetModel2InfoFile= codecs.open(logsPath + "/" + parameters["name"] + "_model2_Predictions.pkl", mode= "w", encoding= "utf-8")
        
        predictions= currentMode2InfoFileContent["predictions"]
        predictions.append(np.asarray(classifier2Predictions))
        targetGoldStandard= currentMode2InfoFileContent["targetGoldStandard"]
        targetGoldStandard.append(np.asarray(completeDataset.target))
        dataInstances= currentMode2InfoFileContent["dataInstances"]
        dataInstances.append(np.asarray(completeDataset.data))
        confusionMatrixes= currentMode2InfoFileContent["confusionMatrixes"]
        confusionMatrixes.append(np.asarray(model2ConfMatrixCurrentIteration))
        
        pickle.dump({
            "predictions": predictions,
            "targetGoldStandard": targetGoldStandard,
            "dataInstances": dataInstances,
            "confusionMatrixes": confusionMatrixes
            }, 
            testSetModel2InfoFile)
        
        testSetModel2InfoFile.close()
        
        # update model2 best model
        if os.path.isfile(modelsPath + "/" + parameters["name"] + "_best_model2.h5"):
            
            if currentModel2Score > bestIterationPredictionsScoreModel2:
                print "found best classifier for target dataset at iteration " + str(iterId)
                
                classifier2Model.save_weights(filepath= modelsPath + "/" + parameters["name"] + "_best_model2.h5")
                
                print "model saved successfully!"
                
                bestIterationPredictionsScoreModel2= currentModel2Score
            
        else:
            # save model weights after the first iteration
            classifier2Model.save_weights(filepath= modelsPath + "/" + parameters["name"] + "_best_model2.h5")
            
        
        
    
    
    print "\n\n +++++   FINAL SCORES   +++++\n\n"
    
    print "\n\n Model 1 - Final Confusion Matrix - Test Set: \n"
    print np.sum(model1ConfMatrixByIteration, axis=0)
    
    print "\n\n Model 2 - Final Confusion Matrix:\n"
    print np.sum(model2ConfMatrixByIteration, axis=0)
    
    sys.stdout = sys.__stdout__
    
    

