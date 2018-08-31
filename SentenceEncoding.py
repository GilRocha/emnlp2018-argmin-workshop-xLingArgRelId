#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentence Encoding
"""

import os
import pickle
import numpy as np
import codecs
import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Paths and filenames
multilingualWordEmbeddingsPath= os.path.abspath("data/WordEmbeddings/cmu")


class SentenceEncoding():
    
    def __init__(self, dataset, language= "en", name= "conventional", wordEmbeddingsDimensions= 300, maxNumberOfTokensInSequence= 50, experimentName= "experiment", datasetPath= os.path.abspath("data/generatedDatasets/en/essays/"), datasetFilename= "ArgEssaysCorpus_en"):
        
        self.name= name
        
        self.datasetPath= datasetPath
        
        self.datasetFilename= datasetFilename
        
        self.wordEmbeddingsDict= None
        
        self.wordEmbeddingsDimensions= wordEmbeddingsDimensions
        
        self.maxNumberOfTokensInSequence= maxNumberOfTokensInSequence
        
        self.tokenizer= None
        
        self.embeddingsMatrixUpdated= False
        
        self.experimentName= experimentName
        
        self.language= language
        
        self.createEmbeddingsMatrix(dataset, self.language)
        
        wordEmbeddingsMatrixFile= codecs.open(self.datasetPath + "/" + "WordEmbeddings" + "/" + self.datasetFilename + "_EmbeddingsMatrix" + ".pkl", mode= "r")
        
        self.embeddingsMatrix= pickle.load(wordEmbeddingsMatrixFile)
        
        wordEmbeddingsMatrixFile.close()
        
    
    
    def transform(self, X):
        
        self.premiseEncoding= []
        self.claimEncoding= []
        
        for learningInstance in X:
            
            # sentence tokens separated by a white space
            # uses the tokenization performed in the preprocessing step and outputs the content in a proper format to be used in the next steps of the process
            premiseTokensSeq= ' '.join([token for token in text_to_word_sequence(learningInstance["SourceADU"])]) #' '.join([token for token in learningInstance["SourceADU_tokens"]])
            
            if self.name == "conventional":
                claimTokensSeq= ' '.join([token for token in text_to_word_sequence(learningInstance["TargetADU"])]) #' '.join([token for token in learningInstance["TargetADU_tokens"]])
            elif self.name == "specialCharBeginOfClaim":
                claimTokensSeq= ' '.join(["delim"] + [token for token in text_to_word_sequence(learningInstance["TargetADU"])]) #' '.join(["delim"] + [token for token in learningInstance["TargetADU_tokens"]])
            
            
            (self.premiseEncoding).append(premiseTokensSeq)
            (self.claimEncoding).append(claimTokensSeq)
        
        
        if self.tokenizer is None:
            
            # Input Layers
            self.tokenizer= Tokenizer(oov_token= "unk")
            
            
            (self.tokenizer).word_index= {}
            
            wordEmbeddingsDictFile= codecs.open(self.datasetPath + "/" + "WordEmbeddings" +  "/" + self.datasetFilename + "_EmbeddingsDictionary" + ".pkl", mode= "r")
            
            self.wordEmbeddingsDict= pickle.load(wordEmbeddingsDictFile)
            
            wordEmbeddingsDictFile.close()
            
            (self.tokenizer).word_index= self.wordEmbeddingsDict
            
        
        # Lowest index in tokenizer.word_index is 1 -> the index 0 is reserved for truncation during padding (add zeros if "maxNumberOfTokensInSequence" > sentence length)
        # in this setting the vocabulary already includes the special chars: delim (special char to signal begin of discourse unit) and unk (for oov words)
        # since after padding the sentence may include zeros, we should consider this special char in the vocabulary -> therefore, we add +1 to the vocab size 
        self.vocabularySize = len((self.tokenizer).word_index) + 1
        print ">>> vocabulary size= " + str(self.vocabularySize)
        
        
        premiseInputSequences = (self.tokenizer).texts_to_sequences(self.premiseEncoding)
        claimInputSequences= (self.tokenizer).texts_to_sequences(self.claimEncoding)
        
        premiseInputSequencesAferPadding = pad_sequences(premiseInputSequences, maxlen= self.maxNumberOfTokensInSequence, padding= "post", truncating= "post")
        claimInputSequencesAferPadding = pad_sequences(claimInputSequences, maxlen= self.maxNumberOfTokensInSequence, padding= "post", truncating= "post")
        
        return [np.asarray(premiseInputSequencesAferPadding), np.asarray(claimInputSequencesAferPadding)]
        
        
        
    
    
    def getMultilingualEmbeddings(self, filepath, filename):
        
        return KeyedVectors.load_word2vec_format(filepath + "/" + filename + ".txt", binary= False)
    
    def getClosestWordsTranslated(self, multiLingualWordEmbeddingsDict, sourceLanguageAcronym, sourceWord, targetLanguageAcronym, nWords= 1):
        
        sourceWordEmbeddingVector= None
        
        if sourceWord in multiLingualWordEmbeddingsDict[sourceLanguageAcronym].wv:
            sourceWordEmbeddingVector= multiLingualWordEmbeddingsDict[sourceLanguageAcronym].wv[sourceWord]
        else:
            return None
        
        return multiLingualWordEmbeddingsDict[targetLanguageAcronym].similar_by_vector(sourceWordEmbeddingVector, topn= nWords)
    
    def getClosestWordsSourceLanguage(self, multiLingualWordEmbeddingsDict, sourceLanguageAcronym, targetLanguageWord, targetLanguageAcronym, nWords= 1):
        
        targetWordEmbeddingVector= None
        
        if targetLanguageWord in multiLingualWordEmbeddingsDict[targetLanguageAcronym].wv:
            targetWordEmbeddingVector= multiLingualWordEmbeddingsDict[targetLanguageAcronym].wv[targetLanguageWord]
        else:
            return None
        
        return multiLingualWordEmbeddingsDict[sourceLanguageAcronym].similar_by_vector(targetWordEmbeddingVector, topn= nWords)
    
    
    def createEmbeddingsMatrix(self, dataset, language= "en"):
        
        if (os.path.isfile(self.datasetPath + "/" + "WordEmbeddings" + "/" + self.datasetFilename + "_EmbeddingsDictionary" + ".pkl")) and (os.path.isfile(self.datasetPath + "/" + "WordEmbeddings" + "/" + self.datasetFilename + "_EmbeddingsMatrix" + ".pkl")):
            print "Embeddings files already exist for the target dataset! Skipping creation of embeddings matrix and dictionary. Existing resources will be used!"
            return True
        
        mulilingualEmbeddingsFilename= "multilingualEmbeddings_" + language
        
        tokenizer= Tokenizer(oov_token= "unk")
        
        premiseEncoding= []
        claimEncoding= []
        
        for elem in dataset.data:
            premiseTokensSeq= ' '.join([token for token in text_to_word_sequence(elem["SourceADU"])])
            
            premiseEncoding.append(premiseTokensSeq)
            
            claimTokensSeq= []
            
            
            if self.name == "conventional":
                claimTokensSeq= ' '.join([token for token in text_to_word_sequence(elem["TargetADU"])])
            elif self.name == "specialCharBeginOfClaim":
                claimTokensSeq= ' '.join(["delim"] + [token for token in text_to_word_sequence(elem["TargetADU"])])
            
            if len(claimTokensSeq) == 0:
                raise Exception("Sequence of claim tokens is empty! Invalid SentenceEncoding.name?")
            
            claimEncoding.append(claimTokensSeq)
        
        
        sentences= premiseEncoding + claimEncoding
        tokenizer.fit_on_texts(sentences)
        
        
        wordEmbeddings= self.getMultilingualEmbeddings(multilingualWordEmbeddingsPath, mulilingualEmbeddingsFilename)
        
        vocabularySize = len((tokenizer).word_index) + 1
        print ">>> vocabulary size= " + str(vocabularySize)
        
        
        
        # Embeddings
        
        embeddingsMatrix = np.zeros((vocabularySize, self.wordEmbeddingsDimensions))
        
        
        for word, i in ((tokenizer).word_index).items():
            
            if word in (wordEmbeddings).wv:
                # words not found in embedding index will be all-zeros.
                embeddingsMatrix[i] = wordEmbeddings.wv[word]
            
        
        wordEmbeddingsDictFile= codecs.open(self.datasetPath + "/" + "WordEmbeddings" + "/" + self.datasetFilename + "_EmbeddingsDictionary" + ".pkl", mode= "w", encoding= "utf-8")
        
        pickle.dump(tokenizer.word_index, wordEmbeddingsDictFile)
        
        wordEmbeddingsDictFile.close()
        
        
        wordEmbeddingsDictFile= codecs.open(self.datasetPath + "/" + "WordEmbeddings" + "/" + self.datasetFilename + "_EmbeddingsMatrix" + ".pkl", mode= "w", encoding= "utf-8")
        
        pickle.dump(embeddingsMatrix, wordEmbeddingsDictFile)
        
        wordEmbeddingsDictFile.close()
        
        return True
    
    
