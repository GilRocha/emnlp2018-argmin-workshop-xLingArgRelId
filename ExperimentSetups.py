#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Setup experiments for Argumentative Relation Identification
"""


import os
import codecs
import sys

import ArgRelationIdentification

# Paths
logsPath= os.path.abspath("data/logs")



# In-Language Experiments
# Runs cross-validation (CV) experiments for a specific Neural Network Architecture
# Assumes that a pickled file with the CV splits exists (same name as parameters["name"])
def inLanguageLearning(parameters= None):
    
    sys.stdout= codecs.open(logsPath + "/" + str(parameters["name"]) + "_console.txt", 'w', encoding= "utf-8")
    
    myModel= ArgRelationIdentification.ArgumentativeRelationClassifier(parameters["datasetPath"], parameters["datasetFilename"], parameters["datasetLanguage"])
    
    resultsDict= myModel.singleLanguageLearn(
        myModel.completeDataset, 
        experimentName= parameters["name"],
        modelFilename= parameters["name"], 
        numberOfClasses= 2, 
        numberOfSplits= parameters["nSplits"], 
        sentenceEncodingType= parameters["sentenceEncodingType"], 
        neuralNetArchitectureName= parameters["neuralNetArchitectureName"],
        balanceDataset= parameters["balanceDataset"],
        classWeight= parameters["classWeight"])
    
    print "\n### Training Set - Evaluation metrics ###"
    print resultsDict["trainingSet"]["finalConfusionMatrix"]
    
    print "\n### Validation Set - Evaluation metrics ###"
    print resultsDict["validationSet"]["finalConfusionMatrix"]
    
    print "\n### Test Set - Evaluation metrics ###"
    #myModel.evaluationMetrics(completeDataset.target, resultsDict["testSet"]["predictions"], completeDataset.target_names)
    print resultsDict["testSet"]["finalConfusionMatrix"]
    
    
    sys.stdout = sys.__stdout__

# In-Language Experiments
# Runs cross-validation (CV) experiments for a Feature-based ML approach
# Assumes that a pickled file with the CV splits exists (same name as parameters["name"])
def inLanguageLearning_FeatureBased(parameters= None):
    
    sys.stdout= codecs.open(logsPath + "/" + str(parameters["name"]) + "_console.txt", 'w', encoding= "utf-8")
    
    myModel= ArgRelationIdentification.ArgumentativeRelationClassifier(parameters["datasetPath"], parameters["datasetFilename"], parameters["datasetLanguage"])
    
    resultsDict= myModel.singleLanguageLearn_FeatureBasedApproach(
        myModel.completeDataset,
        experimentName= parameters["name"],
        modelFilename= parameters["name"], 
        numberOfClasses= 2, 
        numberOfSplits= parameters["nSplits"], 
        balanceDataset= parameters["balanceDataset"],
        classWeight= parameters["classWeight"])
    
    print "\n### Training Set - Evaluation metrics ###"
    print resultsDict["trainingSet"]["finalConfusionMatrix"]
    
    print "\n### Validation Set - Evaluation metrics ###"
    print resultsDict["validationSet"]["finalConfusionMatrix"]
    
    print "\n### Test Set - Evaluation metrics ###"
    #myModel.evaluationMetrics(completeDataset.target, resultsDict["testSet"]["predictions"], completeDataset.target_names)
    print resultsDict["testSet"]["finalConfusionMatrix"]
    
    
    sys.stdout = sys.__stdout__

# In-Language Experiments
# Single run of a specific Neural Network Architecture
def inLanguageLearning_SingleRun(parameters= None):
    
    sys.stdout= codecs.open(logsPath + "/" + str(parameters["name"]) + "_console.txt", 'w', encoding= "utf-8")
    
    myModel= ArgRelationIdentification.ArgumentativeRelationClassifier(parameters["datasetPath"], parameters["datasetFilename"], parameters["datasetLanguage"])
    
    
    myModel.learn(
        experimentName= parameters["name"],
        modelFilename= parameters["name"] + "_model", 
        numberOfClasses= 2, 
        sentenceEncodingType= parameters["sentenceEncodingType"],
        neuralNetArchitectureName= parameters["neuralNetArchitectureName"],
        loadModelFilename= None,
        balanceDataset= parameters["balanceDataset"],
        classWeight= parameters["classWeight"])
    
    predictionsDict= {"neuralNetArchitectureName": parameters["neuralNetArchitectureName"], "numberOfClasses": 2, "modelFilename": str(parameters["name"]) + "_model" + ".h5"}
    preds= myModel.getPredictions(myModel.testSet, predictionsDict)
    
    myModel.evaluationMetrics(myModel.testSet.target, preds, myModel.testSet.target_names)
    
    
    sys.stdout = sys.__stdout__

# Encapsulates several runs (each with different setups e.g. NN architecture) of cross-validation in-language experiments
def runSeveralExperiments_InLanguage(sentenceEncodingType= "conventional", balanceDataset= False, classWeight= False, language= "en"):
    
    nameSuffix= "_" + sentenceEncodingType + "_"
    
    if balanceDataset:
        nameSuffix= nameSuffix + "RandomUndersampling_" 
    
    if classWeight:
        nameSuffix= nameSuffix + "ClassWeight_"
    
    nameSuffix = nameSuffix + language
    
    datasetPath= None
    datasetFilename= None
    
    if language == "en":
        datasetPath= os.path.abspath("data/generatedDatasets/en/essays/")
        datasetFilename= "ArgEssaysCorpus_context_en"
    elif language == "pt":
        datasetPath= os.path.abspath("data/generatedDatasets/pt/")
        datasetFilename= "ArgMineCorpus_context_pt"
    else:
        raise Exception("Specified language not supported.")
    
    
    inLanguageLearning({
        "name": "SumOfEmbeddings_Concatenation_1Layer" + nameSuffix,
        "datasetPath": datasetPath,
        "datasetFilename":datasetFilename,
        "datasetLanguage": language,
        "balanceDataset": balanceDataset,
        "classWeight": classWeight,
        "sentenceEncodingType": sentenceEncodingType,
        "neuralNetArchitectureName": "SumOfEmbeddings_Concatenation_1Layer",
        "nSplits": 5
    })
    
    inLanguageLearning({
        "name": "Concatenate_BiLSTM_1Layer" + nameSuffix,
        "datasetPath": datasetPath,
        "datasetFilename":datasetFilename,
        "datasetLanguage": language,
        "balanceDataset": balanceDataset,
        "classWeight": classWeight,
        "sentenceEncodingType": sentenceEncodingType,
        "neuralNetArchitectureName": "Concatenate_BiLSTM_1Layer",
        "nSplits": 5
    })
    
    
    inLanguageLearning({
        "name": "ClaimAttentionOnPremise" + nameSuffix,
        "datasetPath": datasetPath,
        "datasetFilename":datasetFilename,
        "datasetLanguage": language,
        "balanceDataset": balanceDataset,
        "classWeight": classWeight,
        "sentenceEncodingType": sentenceEncodingType,
        "neuralNetArchitectureName": "ClaimAttentionOnPremise",
        "nSplits": 5
    })
    
    """
    inLanguageLearning({
        "name": "Conv1D_Concatenation_1Layer" + nameSuffix,
        "datasetPath": datasetPath,
        "datasetFilename":datasetFilename,
        "datasetLanguage": language,
        "balanceDataset": balanceDataset,
        "classWeight": classWeight,
        "sentenceEncodingType": sentenceEncodingType,
        "neuralNetArchitectureName": "Conv1D_Concatenation_1Layer",
        "nSplits": 5
    })
    
    inLanguageLearning({
        "name": "Concatenate_LSTM_1Layer" + nameSuffix,
        "datasetPath": datasetPath,
        "datasetFilename":datasetFilename,
        "datasetLanguage": language,
        "balanceDataset": balanceDataset,
        "classWeight": classWeight,
        "sentenceEncodingType": sentenceEncodingType,
        "neuralNetArchitectureName": "Concatenate_LSTM_1Layer",
        "nSplits": 5
    })
    """
    
    """
    
    inLanguageLearning({
        "name": "LSTM_Concatenation_1Layer" + nameSuffix,
        "datasetPath": datasetPath,
        "datasetFilename":datasetFilename,
        "datasetLanguage": language,
        "balanceDataset": balanceDataset,
        "classWeight": classWeight,
        "sentenceEncodingType": sentenceEncodingType,
        "neuralNetArchitectureName": "LSTM_Concatenation_1Layer",
        "nSplits": 5
    })
    
    inLanguageLearning({
        "name": "ClaimAttentionOnPremise_1Layer" + nameSuffix,
        "datasetPath": datasetPath,
        "datasetFilename":datasetFilename,
        "datasetLanguage": language,
        "balanceDataset": balanceDataset,
        "classWeight": classWeight,
        "sentenceEncodingType": sentenceEncodingType,
        "neuralNetArchitectureName": "ClaimAttentionOnPremise_1Layer",
        "nSplits": 5
    })
    
    inLanguageLearning({
        "name": "SharedLSTM_Concatenation_1Layer" + nameSuffix,
        "datasetPath": datasetPath,
        "datasetFilename":datasetFilename,
        "datasetLanguage": language,
        "balanceDataset": balanceDataset,
        "classWeight": classWeight,
        "sentenceEncodingType": sentenceEncodingType,
        "neuralNetArchitectureName": "SharedLSTM_Concatenation_1Layer",
        "nSplits": 5
    })
    """
    
    
    

# Encapsulates several runs (each with different setups e.g. NN architecture) of Direct Transfer experiments (each running n times with different seeds)
def runSeveralExperiments_TransferLearning(params= None):
    
    nameSuffix= "_" + params["sentenceEncodingType"] + "_"
    
    if params["balanceDataset"]:
        nameSuffix= nameSuffix + "RandomUndersampling_" 
    
    if params["classWeight"]:
        nameSuffix= nameSuffix + "ClassWeight_"
    
    nameSuffix = nameSuffix + params["dataset1Language"] + "_" + params["dataset2Language"]
    
    
    ArgRelationIdentification.directTransferLearning({
        "name": "SumOfEmbeddings_Concatenation_1Layer" + nameSuffix,
        "dataset1Path": params["dataset1Path"], 
        "dataset1Filename": params["dataset1Filename"], 
        "dataset1Language": params["dataset1Language"],
        "dataset2Path": params["dataset2Path"], 
        "dataset2Filename": params["dataset2Filename"],
        "dataset2Language": params["dataset2Language"],
        "balanceDataset": params["balanceDataset"],
        "classWeight": params["classWeight"],
        "sentenceEncodingType": params["sentenceEncodingType"],
        "neuralNetArchitectureName": "SumOfEmbeddings_Concatenation_1Layer"
    })
    
    ArgRelationIdentification.directTransferLearning({
        "name": "ClaimAttentionOnPremise" + nameSuffix,
        "dataset1Path": params["dataset1Path"], 
        "dataset1Filename": params["dataset1Filename"], 
        "dataset1Language": params["dataset1Language"],
        "dataset2Path": params["dataset2Path"], 
        "dataset2Filename": params["dataset2Filename"],
        "dataset2Language": params["dataset2Language"],
        "balanceDataset": params["balanceDataset"],
        "classWeight": params["classWeight"],
        "sentenceEncodingType": params["sentenceEncodingType"],
        "neuralNetArchitectureName": "ClaimAttentionOnPremise"
    })
    
    ArgRelationIdentification.directTransferLearning({
        "name": "Concatenate_BiLSTM_1Layer" + nameSuffix,
        "dataset1Path": params["dataset1Path"], 
        "dataset1Filename": params["dataset1Filename"], 
        "dataset1Language": params["dataset1Language"],
        "dataset2Path": params["dataset2Path"], 
        "dataset2Filename": params["dataset2Filename"],
        "dataset2Language": params["dataset2Language"],
        "balanceDataset": params["balanceDataset"],
        "classWeight": params["classWeight"],
        "sentenceEncodingType": params["sentenceEncodingType"],
        "neuralNetArchitectureName": "Concatenate_BiLSTM_1Layer"
    })
    
    
    
    """
    ArgRelationIdentification.directTransferLearning({
        "name": "Conv1D_Concatenation_1Layer" + nameSuffix,
        "dataset1Path": params["dataset1Path"], 
        "dataset1Filename": params["dataset1Filename"], 
        "dataset1Language": params["dataset1Language"],
        "dataset2Path": params["dataset2Path"], 
        "dataset2Filename": params["dataset2Filename"],
        "dataset2Language": params["dataset2Language"],
        "balanceDataset": params["balanceDataset"],
        "classWeight": params["classWeight"],
        "sentenceEncodingType": params["sentenceEncodingType"],
        "neuralNetArchitectureName": "Conv1D_Concatenation_1Layer"
    })
    
    ArgRelationIdentification.directTransferLearning({
        "name": "Concatenate_LSTM_1Layer" + nameSuffix,
        "dataset1Path": params["dataset1Path"], 
        "dataset1Filename": params["dataset1Filename"], 
        "dataset1Language": params["dataset1Language"],
        "dataset2Path": params["dataset2Path"], 
        "dataset2Filename": params["dataset2Filename"],
        "dataset2Language": params["dataset2Language"],
        "balanceDataset": params["balanceDataset"],
        "classWeight": params["classWeight"],
        "sentenceEncodingType": params["sentenceEncodingType"],
        "neuralNetArchitectureName": "Concatenate_LSTM_1Layer"
    })
    """
    
    """
    ArgRelationIdentification.directTransferLearning({
        "name": "LSTM_Concatenation_1Layer" + nameSuffix,
        "dataset1Path": params["dataset1Path"], 
        "dataset1Filename": params["dataset1Filename"], 
        "dataset1Language": params["dataset1Language"],
        "dataset2Path": params["dataset2Path"], 
        "dataset2Filename": params["dataset2Filename"],
        "dataset2Language": params["dataset2Language"],
        "balanceDataset": params["balanceDataset"],
        "classWeight": params["classWeight"],
        "sentenceEncodingType": params["sentenceEncodingType"],
        "neuralNetArchitectureName": "LSTM_Concatenation_1Layer"
    })
    
    """
    
    
    
    """
    ArgRelationIdentification.directTransferLearning({
        "name": "SharedLSTM_Concatenation_1Layer" + nameSuffix,
        "dataset1Path": params["dataset1Path"], 
        "dataset1Filename": params["dataset1Filename"], 
        "dataset1Language": params["dataset1Language"],
        "dataset2Path": params["dataset2Path"], 
        "dataset2Filename": params["dataset2Filename"],
        "dataset2Language": params["dataset2Language"],
        "balanceDataset": params["balanceDataset"],
        "classWeight": params["classWeight"],
        "sentenceEncodingType": params["sentenceEncodingType"],
        "neuralNetArchitectureName": "SharedLSTM_Concatenation_1Layer"
    })
    
    ArgRelationIdentification.directTransferLearning({
        "name": "ClaimAttentionOnPremise_1Layer" + nameSuffix,
        "dataset1Path": params["dataset1Path"], 
        "dataset1Filename": params["dataset1Filename"], 
        "dataset1Language": params["dataset1Language"],
        "dataset2Path": params["dataset2Path"], 
        "dataset2Filename": params["dataset2Filename"],
        "dataset2Language": params["dataset2Language"],
        "balanceDataset": params["balanceDataset"],
        "classWeight": params["classWeight"],
        "sentenceEncodingType": params["sentenceEncodingType"],
        "neuralNetArchitectureName": "ClaimAttentionOnPremise_1Layer"
    })
    """
    
    





