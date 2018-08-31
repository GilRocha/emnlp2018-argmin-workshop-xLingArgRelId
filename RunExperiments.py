#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Run experiments utils
"""


import os
import codecs
import sys

import ArgRelationIdentification
import ExperimentSetups



#######################################
#####   In-language Experiments   #####
#######################################

# Cross-Validation
# Neural Network Architectures

"""
# ArgEssays EN
ExperimentSetups.runSeveralExperiments_InLanguage(sentenceEncodingType= "specialCharBeginOfClaim", balanceDataset= False, classWeight= False, language= "en")
ExperimentSetups.runSeveralExperiments_InLanguage(sentenceEncodingType= "specialCharBeginOfClaim", balanceDataset= False, classWeight= True, language= "en")
ExperimentSetups.runSeveralExperiments_InLanguage(sentenceEncodingType= "specialCharBeginOfClaim", balanceDataset= True, classWeight= False, language= "en")
"""

"""
# ArgMine PT
ExperimentSetups.runSeveralExperiments_InLanguage(sentenceEncodingType= "specialCharBeginOfClaim", balanceDataset= False, classWeight= False, language= "pt")
ExperimentSetups.runSeveralExperiments_InLanguage(sentenceEncodingType= "specialCharBeginOfClaim", balanceDataset= False, classWeight= True, language= "pt")
ExperimentSetups.runSeveralExperiments_InLanguage(sentenceEncodingType= "specialCharBeginOfClaim", balanceDataset= True, classWeight= False, language= "pt")
"""




# Cross-Validation
# Feature-based approaches

"""
# ArgEssays EN
ExperimentSetups.inLanguageLearning_FeatureBased({
        "name": "NGram_LogReg_Balanced_en",
        "datasetPath": os.path.abspath("data/generatedDatasets/en/essays/"),
        "datasetFilename":"ArgEssaysCorpus_context_en",
        "datasetLanguage": "en",
        "balanceDataset": False,
        "classWeight": False,
        "nSplits": 5
    })


# ArgMine PT
ExperimentSetups.inLanguageLearning_FeatureBased({
        "name": "NGram_LogReg_Balanced_pt",
        "datasetPath": os.path.abspath("data/generatedDatasets/pt/"),
        "datasetFilename":"ArgMineCorpus_context_pt",
        "datasetLanguage": "pt",
        "balanceDataset": False,
        "classWeight": False,
        "nSplits": 5
    })
"""



###########################################
#####   Direct Transfer Experiments   #####
###########################################


"""
# Direct Transfer -> Arg Essays EN to ArgMine PT
ExperimentSetups.runSeveralExperiments_TransferLearning({
    "dataset1Path": os.path.abspath("data/generatedDatasets/en/essays/"), 
    "dataset1Filename": "ArgEssaysCorpus_context_en", 
    "dataset1Language": "en",
    "dataset2Path": os.path.abspath("data/generatedDatasets/pt/"), 
    "dataset2Filename": "ArgMineCorpus_context_pt",
    "dataset2Language": "pt",
    "sentenceEncodingType": "specialCharBeginOfClaim", 
    "balanceDataset": False, 
    "classWeight": False
    })
    

ExperimentSetups.runSeveralExperiments_TransferLearning({
    "dataset1Path": os.path.abspath("data/generatedDatasets/en/essays/"), 
    "dataset1Filename": "ArgEssaysCorpus_context_en", 
    "dataset1Language": "en",
    "dataset2Path": os.path.abspath("data/generatedDatasets/pt/"), 
    "dataset2Filename": "ArgMineCorpus_context_pt",
    "dataset2Language": "pt",
    "sentenceEncodingType": "specialCharBeginOfClaim", 
    "balanceDataset": False, 
    "classWeight": True
    })
"""    


"""
# Projection approach ->  ArgEssays EN translated to PT 
ExperimentSetups.runSeveralExperiments_TransferLearning({
    "dataset1Path": os.path.abspath("data/generatedDatasets/en/essays/"), 
    "dataset1Filename": "ArgEssaysCorpus_context_en_translatedTo_pt", 
    "dataset1Language": "pt",
    "dataset2Path": os.path.abspath("data/generatedDatasets/pt/"), 
    "dataset2Filename": "ArgMineCorpus_context_pt",
    "dataset2Language": "pt",
    "sentenceEncodingType": "specialCharBeginOfClaim", 
    "balanceDataset": False, 
    "classWeight": False
    })

ExperimentSetups.runSeveralExperiments_TransferLearning({
    "dataset1Path": os.path.abspath("data/generatedDatasets/en/essays/"), 
    "dataset1Filename": "ArgEssaysCorpus_context_en_translatedTo_pt", 
    "dataset1Language": "pt",
    "dataset2Path": os.path.abspath("data/generatedDatasets/pt/"), 
    "dataset2Filename": "ArgMineCorpus_context_pt",
    "dataset2Language": "pt",
    "sentenceEncodingType": "specialCharBeginOfClaim", 
    "balanceDataset": False, 
    "classWeight": True
    })

ExperimentSetups.runSeveralExperiments_TransferLearning({
    "dataset1Path": os.path.abspath("data/generatedDatasets/en/essays/"), 
    "dataset1Filename": "ArgEssaysCorpus_context_en_translatedTo_pt", 
    "dataset1Language": "pt",
    "dataset2Path": os.path.abspath("data/generatedDatasets/pt/"), 
    "dataset2Filename": "ArgMineCorpus_context_pt",
    "dataset2Language": "pt",
    "sentenceEncodingType": "specialCharBeginOfClaim", 
    "balanceDataset": True, 
    "classWeight": False
    })
"""


