#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Neural Network Architectures for Argumentative Relation Identification
"""


import os
import codecs

from keras import backend as K
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Dropout, concatenate, Input, LSTM, Embedding, Lambda, BatchNormalization, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Reshape, Activation, RepeatVector, Permute, Multiply, Bidirectional, Dot

import pickle

import inner_attention_layer
import ArgRelationIdentification



def neuralNetwork(architectureName= "SumOfEmbeddings_Concatenation_1Layer", numberOfClasses= 2, layerActivation= "relu", dropoutRate= 0.2, embeddingsDimensionalityReduction_FinalDimensions= 100, maxTokensInSentence= 50, embeddingsMatrixPath= None, embeddingsMatrixFilename= None, randomSeedValue= 0):
    
    ArgRelationIdentification.randomSeedInitialization(randomSeedValue)
    
    embeddingsMatrixFile= codecs.open(embeddingsMatrixPath + "/" + embeddingsMatrixFilename + ".pkl", mode= "r")
    
    embeddingsMatrix= pickle.load(embeddingsMatrixFile)
    
    embeddingsMatrixFile.close()
    
    embeddingsVectorDimension= len(embeddingsMatrix[0])
    vocabularySize= len(embeddingsMatrix)
    
    if architectureName == "SumOfEmbeddings_Concatenation_1Layer":
        
        # Input Layers
        premiseInput= Input(shape= (maxTokensInSentence,), name= "Premise_Input")
        claimInput= Input(shape= (maxTokensInSentence,), name= "Claim_Input")
        
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        
        premiseEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Premise_EmbedLayer")
        
        claimEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Claim_EmbedLayer")
        
        premiseEmbeddingsLayer= premiseEmbeddingLayer(premiseInput)
        claimEmbeddingsLayer= claimEmbeddingLayer(claimInput)
        
        # Embeddings Dimensionality Reduction
        premiseReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Premise_ReducedEmbed")(premiseEmbeddingsLayer)
            
        claimReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Claim_ReducedEmbed")(claimEmbeddingsLayer)
        
        """
        premiseEmbedOutDropout= Dropout(dropoutRate, name= "Premise_EncodingDrop")(premiseReducedEmbeddingsLayer)
        claimEmbedOutDropout= Dropout(dropoutRate, name= "Claim_EncodingDrop")(claimReducedEmbeddingsLayer)
        
        premiseEmbedOutNormalization= BatchNormalization(name= "Premise_EncodingNorm")(premiseEmbedOutDropout)
        claimEmbedOutNormalization= BatchNormalization(name= "Claim_EncodingNorm")(claimEmbedOutDropout)
        """
        
        # Sum embeddings of each layer
        # Each layer has shape (numTokens, embsDimension)
        # The resulting structure has shape (embsDimension) and corresponds to the sum of the embeddings in the previous layer
        premiseEmbeddingSum = Lambda(lambda x: K.sum(x, axis= 1), output_shape= (embeddingsDimensionalityReduction_FinalDimensions,), name= "Premise_SumWordEmbed")(premiseReducedEmbeddingsLayer)
        claimEmbeddingSum = Lambda(lambda x: K.sum(x, axis= 1), output_shape= (embeddingsDimensionalityReduction_FinalDimensions,), name= "Claim_SumWordEmbed")(claimReducedEmbeddingsLayer)
        
        """
        premiseEmbedSumNormalization= BatchNormalization(name= "Premise_SumWordNorm")(premiseEmbeddingSum)
        claimEmbedSumNormalization= BatchNormalization(name= "Claim_SumWordNorm")(claimEmbeddingSum)
        """
        
        # We can then concatenate the two vectors:
        mergedSentenceEncodings = concatenate([premiseEmbeddingSum, claimEmbeddingSum], axis=-1, name="Premise_Claim_Concat")
        
        firstHiddenLayer = Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            activation= layerActivation, 
            input_dim= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            name= "Hidden1")(mergedSentenceEncodings)
        
        
        firstDropout= Dropout(dropoutRate, name= "Hidden1Drop")(firstHiddenLayer)
        
        finalNormalization= BatchNormalization(name= "NormLayer")(firstDropout)
        
        softmaxLayer= Dense(units=numberOfClasses, activation='softmax', name= "Softmax")(finalNormalization)
        #softmaxLayer= Dense(units=1, activation='sigmoid', name= "Softmax")(finalNormalization)
        
        sentenceEncodingModel = Model(inputs=[premiseInput, claimInput], outputs=softmaxLayer)
        
        # Configuring neural net learning process
        #sentenceEncodingModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        sentenceEncodingModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Print neural network architecture summary
        sentenceEncodingModel.summary()
        
        return sentenceEncodingModel
        
    elif architectureName == "LSTM_Concatenation_1Layer":
        
        # Input Layers
        premiseInput= Input(shape= (maxTokensInSentence,), name= "Premise_Input")
        claimInput= Input(shape= (maxTokensInSentence,), name= "Claim_Input")
        
        
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        premiseEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Premise_EmbedLayer")
        
        claimEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Claim_EmbedLayer")
        
        
        premiseEmbeddingsLayer= premiseEmbeddingLayer(premiseInput)
        claimEmbeddingsLayer= claimEmbeddingLayer(claimInput)
        
        # Embeddings Dimensionality Reduction
        premiseReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Premise_ReducedEmbed")(premiseEmbeddingsLayer)
            
        claimReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Claim_ReducedEmbed")(claimEmbeddingsLayer)
        
        """
        premiseEmbedOutDropout= Dropout(dropoutRate, name= "Premise_EncodingDrop")(premiseReducedEmbeddingsLayer)
        claimEmbedOutDropout= Dropout(dropoutRate, name= "Claim_EncodingDrop")(claimReducedEmbeddingsLayer)
        
        #premiseEmbedOutNormalization= BatchNormalization(name= "Premise_EncodingNorm")(premiseEmbedOutDropout)
        #claimEmbedOutNormalization= BatchNormalization(name= "Claim_EncodingNorm")(claimEmbedOutDropout)
        """
        
        premiseLSTM = LSTM(units= embeddingsDimensionalityReduction_FinalDimensions, name= "Premise_LSTM")(premiseReducedEmbeddingsLayer)
        claimLSTM = LSTM(units= embeddingsDimensionalityReduction_FinalDimensions, name= "Claim_LSTM")(claimReducedEmbeddingsLayer)
        
        #premiseEmbedSumNormalization= BatchNormalization(name= "Premise_SumWordNorm")(premiseLSTM)
        #claimEmbedSumNormalization= BatchNormalization(name= "Claim_SumWordNorm")(claimLSTM)
        
        # We can then concatenate the two vectors:
        mergedSentenceEncodings = concatenate([premiseLSTM, claimLSTM], axis=-1, name="Premise_Claim_Concat")
        
        firstHiddenLayer = Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            name= "Hidden1")(mergedSentenceEncodings)
        
        firstDropout= Dropout(dropoutRate, name= "Hidden1Drop")(firstHiddenLayer)
        
        finalNormalization= BatchNormalization(name= "NormLayer")(firstDropout)
        
        softmaxLayer= Dense(units=numberOfClasses, activation='softmax', name= "Softmax")(finalNormalization)
        
        sentenceEncodingModel = Model(inputs=[premiseInput, claimInput], outputs=softmaxLayer)
        
        # Configuring neural net learning process
        sentenceEncodingModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        sentenceEncodingModel.summary()
        
        return sentenceEncodingModel
        
    elif architectureName == "SharedLSTM_Concatenation_1Layer":
        
        # Input Layers
        premiseInput= Input(shape= (maxTokensInSentence,), name= "Premise_Input")
        claimInput= Input(shape= (maxTokensInSentence,), name= "Claim_Input")
        
        
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        premiseEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Premise_EmbedLayer")
        
        claimEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Claim_EmbedLayer")
        
        premiseEmbeddingsLayer= premiseEmbeddingLayer(premiseInput)
        claimEmbeddingsLayer= claimEmbeddingLayer(claimInput)
        
        # Embeddings Dimensionality Reduction
        premiseReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Premise_ReducedEmbed")(premiseEmbeddingsLayer)
            
        claimReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Claim_ReducedEmbed")(claimEmbeddingsLayer)
        
        """
        premiseEmbedOutDropout= Dropout(dropoutRate, name= "Premise_EncodingDrop")(premiseReducedEmbeddingsLayer)
        claimEmbedOutDropout= Dropout(dropoutRate, name= "Claim_EncodingDrop")(claimReducedEmbeddingsLayer)
        
        #premiseEmbedOutNormalization= BatchNormalization(name= "Premise_EncodingNorm")(premiseEmbedOutDropout)
        #claimEmbedOutNormalization= BatchNormalization(name= "Claim_EncodingNorm")(claimEmbedOutDropout)
        """
        
        sharedLSTM = LSTM(units= embeddingsDimensionalityReduction_FinalDimensions, name= "SharedLSTM", return_sequences= True, return_state= True)
        
        premiseLSTMEncoding, premiseState_h, premiseState_c= sharedLSTM(premiseReducedEmbeddingsLayer)
        claimLSTMEncoding, claimState_h, claimState_c= sharedLSTM(claimReducedEmbeddingsLayer)
        
        #premiseEmbedSumNormalization= BatchNormalization(name= "Premise_SumWordNorm")(premiseLSTM)
        #claimEmbedSumNormalization= BatchNormalization(name= "Claim_SumWordNorm")(claimLSTM)
        
        # We can then concatenate the two vectors:
        mergedSentenceEncodings = concatenate([premiseState_h, claimState_h], axis=-1, name="Premise_Claim_Concat")
        
        firstHiddenLayer = Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            activation= layerActivation, 
            input_dim= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            name= "Hidden1")(mergedSentenceEncodings)
        
        firstDropout= Dropout(dropoutRate, name= "Hidden1Drop")(firstHiddenLayer)
        
        finalNormalization= BatchNormalization(name= "NormLayer")(firstDropout)
        
        softmaxLayer= Dense(units=numberOfClasses, activation='softmax', name= "Softmax")(finalNormalization)
        
        sentenceEncodingModel = Model(inputs=[premiseInput, claimInput], outputs=softmaxLayer)
        
        # Configuring neural net learning process
        sentenceEncodingModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        sentenceEncodingModel.summary()
        
        return sentenceEncodingModel
        
    elif architectureName == "Conv1D_Concatenation_1Layer":
        
        kernelSize= 2
        stride= 1
        
        # Input Layers
        premiseInput= Input(shape= (maxTokensInSentence,), name= "Premise_Input")
        claimInput= Input(shape= (maxTokensInSentence,), name= "Claim_Input")
        
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        premiseEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Premise_EmbedLayer")
        
        claimEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Claim_EmbedLayer")
        
        
        premiseEmbeddingsLayer= premiseEmbeddingLayer(premiseInput)
        claimEmbeddingsLayer= claimEmbeddingLayer(claimInput)
        
        # Embeddings Dimensionality Reduction
        premiseReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Premise_ReducedEmbed")(premiseEmbeddingsLayer)
            
        claimReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Claim_ReducedEmbed")(claimEmbeddingsLayer)
        
        """
        premiseEmbedOutDropout= Dropout(dropoutRate, name= "Premise_EncodingDrop")(premiseReducedEmbeddingsLayer)
        claimEmbedOutDropout= Dropout(dropoutRate, name= "Claim_EncodingDrop")(claimReducedEmbeddingsLayer)
        
        #premiseEmbedOutNormalization= BatchNormalization(name= "Premise_EncodingNorm")(premiseEmbedOutDropout)
        #claimEmbedOutNormalization= BatchNormalization(name= "Claim_EncodingNorm")(claimEmbedOutDropout)
        """
        
        # Convolutional layer #1
        premiseEncodingAfterConv1D= Conv1D(
            filters= embeddingsDimensionalityReduction_FinalDimensions,
            kernel_size= kernelSize, 
            strides= stride,
            name= "Premise_Conv1D_1")(premiseReducedEmbeddingsLayer)
        
        claimEncodingAfterConv1D= Conv1D(
            filters= embeddingsDimensionalityReduction_FinalDimensions, 
            kernel_size= kernelSize, 
            strides= stride,
            name= "Claim_Conv1D_1")(claimReducedEmbeddingsLayer)
        
        premisePooling= MaxPooling1D(pool_size= maxTokensInSentence - (kernelSize - 1), name= "Premise_MaxPool1D_1")(premiseEncodingAfterConv1D)
        claimPooling= MaxPooling1D(pool_size= maxTokensInSentence - (kernelSize - 1), name= "Claim_MaxPool1D_1")(claimEncodingAfterConv1D)
        
        premisePoolingFlatten= Flatten(name= "PremiseFlatten")(premisePooling)
        claimPoolingFlatten= Flatten(name= "ClaimFlatten")(claimPooling)
        
        #premiseEmbedSumNormalization= BatchNormalization(name= "Premise_SumWordNorm")(premiseLSTM)
        #claimEmbedSumNormalization= BatchNormalization(name= "Claim_SumWordNorm")(claimLSTM)
        
        
        # We can then concatenate the two vectors:
        mergedSentenceEncodings = concatenate([premisePoolingFlatten, claimPoolingFlatten], axis=-1, name="Premise_Claim_Concat")
        
        firstHiddenLayer = Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            activation= layerActivation, 
            input_dim= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            name= "Hidden1")(mergedSentenceEncodings)
        
        firstDropout= Dropout(dropoutRate, name= "Hidden1Drop")(firstHiddenLayer)
        
        finalNormalization= BatchNormalization(name= "NormLayer")(firstDropout)
        
        softmaxLayer= Dense(units=numberOfClasses, activation='softmax', name= "Softmax")(finalNormalization)
        
        sentenceEncodingModel = Model(inputs=[premiseInput, claimInput], outputs=softmaxLayer)
        
        # Configuring neural net learning process
        sentenceEncodingModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        sentenceEncodingModel.summary()
        
        return sentenceEncodingModel
        
    elif architectureName == "ClaimAttentionOnPremise_1Layer":
        
        # Input Layers
        premiseInput= Input(shape= (maxTokensInSentence,), name= "Premise_Input")
        claimInput= Input(shape= (maxTokensInSentence,), name= "Claim_Input")
        
        
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        premiseEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Premise_EmbedLayer")
        
        claimEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Claim_EmbedLayer")
        
        premiseEmbeddingsLayer= premiseEmbeddingLayer(premiseInput)
        claimEmbeddingsLayer= claimEmbeddingLayer(claimInput)
        
        # Embeddings Dimensionality Reduction
        premiseReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Premise_ReducedEmbed")(premiseEmbeddingsLayer)
            
        claimReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Claim_ReducedEmbed")(claimEmbeddingsLayer)
        
        """
        premiseEmbedOutDropout= Dropout(dropoutRate, name= "Premise_EncodingDrop")(premiseReducedEmbeddingsLayer)
        claimEmbedOutDropout= Dropout(dropoutRate, name= "Claim_EncodingDrop")(claimReducedEmbeddingsLayer)
        """
        
        claimLSTM = LSTM(units= embeddingsDimensionalityReduction_FinalDimensions, name= "Claim_LSTM", return_sequences= True, return_state= True)
        claimLSTMEncoding, claimState_h, claimState_c= claimLSTM(claimReducedEmbeddingsLayer)
        
        # Attention mechanism
        
        # very simple softmax (because it will be computed several times) to compute attention weights
        #attention = Dense(1, activation= 'tanh')(claimState_h)
        attention = Dense(1, activation= 'tanh', name= "AttentionDecisionNode")(premiseReducedEmbeddingsLayer)
        attention = Flatten(name= "flatten")(attention)
        attention = Activation('softmax', name= "AttentionActivation")(attention)
        # Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
        attention = RepeatVector(embeddingsDimensionalityReduction_FinalDimensions, name= "WordsAttention")(attention)
        attention = Permute([2, 1], name= "AttentionWeights")(attention)
        
        # compute context vector
        # element-wise multiplication
        contextVector = Multiply(name= "Attention_ContextVector_Mult")([claimState_h, attention])
        # weighted sum
        contextVector = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(embeddingsDimensionalityReduction_FinalDimensions,), name= "Attention_ContextVector")(contextVector)
        
        #premiseEmbedOutNormalization= BatchNormalization(name= "Premise_EncodingNorm")(premiseReducedEmbeddingsLayer)
        #claimEmbedOutNormalization= BatchNormalization(name= "Claim_EncodingNorm")(claimReducedEmbeddingsLayer)
        
        #premiseEmbedSumNormalization= BatchNormalization(name= "Premise_SumWordNorm")(premiseEmbeddingSum)
        #claimEmbedSumNormalization= BatchNormalization(name= "Claim_SumWordNorm")(claimEmbeddingSum)
        
        # We can then concatenate the two vectors:
        mergedSentenceEncodings = concatenate([claimState_h, contextVector], axis=-1, name="ClaimEncod_ContextVector_Concat")
        
        firstHiddenLayer = Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            activation= layerActivation, 
            input_dim= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            name= "Hidden1")(mergedSentenceEncodings)
        
        firstDropout= Dropout(dropoutRate, name= "Hidden1Drop")(firstHiddenLayer)
        
        finalNormalization= BatchNormalization(name= "NormLayer")(firstDropout)
        
        softmaxLayer= Dense(units=numberOfClasses, activation='softmax', name= "Softmax")(finalNormalization)
        
        sentenceEncodingModel = Model(inputs=[premiseInput, claimInput], outputs=softmaxLayer)
        
        # Configuring neural net learning process
        sentenceEncodingModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        sentenceEncodingModel.summary()
        
        return sentenceEncodingModel
        
    elif architectureName == "ClaimAttentionOnPremise":
        # Using Inner Attention Layer developed by Dr. Christian Stab (UKP - TU Darmstadt)
        
        # Input Layers
        premiseInput= Input(shape= (maxTokensInSentence,), name= "Premise_Input")
        claimInput= Input(shape= (maxTokensInSentence,), name= "Claim_Input")
        
        
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        premiseEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Premise_EmbedLayer")
        
        claimEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Claim_EmbedLayer")
        
        premiseEmbeddingsLayer= premiseEmbeddingLayer(premiseInput)
        claimEmbeddingsLayer= claimEmbeddingLayer(claimInput)
        
        # Embeddings Dimensionality Reduction
        premiseReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Premise_ReducedEmbed")(premiseEmbeddingsLayer)
            
        claimReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Claim_ReducedEmbed")(claimEmbeddingsLayer)
        
        """
        premiseEmbedOutDropout= Dropout(dropoutRate, name= "Premise_EncodingDrop")(premiseReducedEmbeddingsLayer)
        claimEmbedOutDropout= Dropout(dropoutRate, name= "Claim_EncodingDrop")(claimReducedEmbeddingsLayer)
        """
        
        premiseLSTM = LSTM(units= embeddingsDimensionalityReduction_FinalDimensions, name= "Premise_LSTM", return_sequences= True, return_state= True)
        premiseLSTMEncoding, premiseState_h, premiseState_c= premiseLSTM(premiseReducedEmbeddingsLayer)
        
        attentationLayer = inner_attention_layer.InnerAttentionLayerKeras(topic=premiseState_h, emb_dim=embeddingsDimensionalityReduction_FinalDimensions, return_sequence=True, name= "AttLayer")(claimReducedEmbeddingsLayer)
        
        attClaimLSTM = LSTM(units= embeddingsDimensionalityReduction_FinalDimensions, name= "AttClaim_LSTM")
        attClaimLSTMEncoding= attClaimLSTM(attentationLayer)
        
        firstHiddenLayer = Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsDimensionalityReduction_FinalDimensions, 
            name= "Hidden1")(attClaimLSTMEncoding)
        
        firstDropout= Dropout(dropoutRate, name= "Hidden1Drop")(firstHiddenLayer)
        
        finalNormalization= BatchNormalization(name= "NormLayer")(firstDropout)
        
        softmaxLayer= Dense(units=numberOfClasses, activation='softmax', name= "Softmax")(finalNormalization)
        
        sentenceEncodingModel = Model(inputs=[premiseInput, claimInput], outputs=softmaxLayer)
        
        # Configuring neural net learning process
        sentenceEncodingModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        sentenceEncodingModel.summary()
        
        return sentenceEncodingModel
        
    elif architectureName == "Concatenate_LSTM_1Layer":
        
        # Input Layers
        premiseInput= Input(shape= (maxTokensInSentence,), name= "Premise_Input")
        claimInput= Input(shape= (maxTokensInSentence,), name= "Claim_Input")
        
        
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        premiseEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Premise_EmbedLayer")
        
        claimEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Claim_EmbedLayer")
        
        premiseEmbeddingsLayer= premiseEmbeddingLayer(premiseInput)
        claimEmbeddingsLayer= claimEmbeddingLayer(claimInput)
        
        # Embeddings Dimensionality Reduction
        premiseReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Premise_ReducedEmbed")(premiseEmbeddingsLayer)
            
        claimReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Claim_ReducedEmbed")(claimEmbeddingsLayer)
        
        """
        premiseEmbedOutDropout= Dropout(dropoutRate, name= "Premise_EncodingDrop")(premiseReducedEmbeddingsLayer)
        claimEmbedOutDropout= Dropout(dropoutRate, name= "Claim_EncodingDrop")(claimReducedEmbeddingsLayer)
        """
        
        mergedSentenceEncodings = concatenate([premiseReducedEmbeddingsLayer, claimReducedEmbeddingsLayer], axis= 1, name="PremiseClaim_Concat")
        
        premiseClaimLSTM = LSTM(units= embeddingsDimensionalityReduction_FinalDimensions, name= "LSTM")
        
        premiseClaimLSTMEncoding= premiseClaimLSTM(mergedSentenceEncodings)
        
        firstHiddenLayer = Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsDimensionalityReduction_FinalDimensions, 
            name= "Hidden1")(premiseClaimLSTMEncoding)
        
        firstDropout= Dropout(dropoutRate, name= "Hidden1Drop")(firstHiddenLayer)
        
        finalNormalization= BatchNormalization(name= "NormLayer")(firstDropout)
        
        softmaxLayer= Dense(units=numberOfClasses, activation='softmax', name= "Softmax")(finalNormalization)
        
        sentenceEncodingModel = Model(inputs=[premiseInput, claimInput], outputs=softmaxLayer)
        
        # Configuring neural net learning process
        sentenceEncodingModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        sentenceEncodingModel.summary()
        
        return sentenceEncodingModel
        
    elif architectureName == "Concatenate_BiLSTM_1Layer":
        
        # Input Layers
        premiseInput= Input(shape= (maxTokensInSentence,), name= "Premise_Input")
        claimInput= Input(shape= (maxTokensInSentence,), name= "Claim_Input")
        
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        premiseEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Premise_EmbedLayer")
        
        claimEmbeddingLayer = Embedding(vocabularySize,
                                    embeddingsVectorDimension,
                                    weights= [embeddingsMatrix],
                                    input_length= maxTokensInSentence,
                                    trainable= False,
                                    name= "Claim_EmbedLayer")
        
        premiseEmbeddingsLayer= premiseEmbeddingLayer(premiseInput)
        claimEmbeddingsLayer= claimEmbeddingLayer(claimInput)
        
        # Embeddings Dimensionality Reduction
        premiseReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Premise_ReducedEmbed")(premiseEmbeddingsLayer)
            
        claimReducedEmbeddingsLayer= Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions, 
            activation= layerActivation, 
            input_dim= embeddingsVectorDimension, 
            name= "Claim_ReducedEmbed")(claimEmbeddingsLayer)
        
        """
        premiseEmbedOutDropout= Dropout(dropoutRate, name= "Premise_EncodingDrop")(premiseReducedEmbeddingsLayer)
        claimEmbedOutDropout= Dropout(dropoutRate, name= "Claim_EncodingDrop")(claimReducedEmbeddingsLayer)
        """
        
        mergedSentenceEncodings = concatenate([premiseReducedEmbeddingsLayer, claimReducedEmbeddingsLayer], axis= 1, name="PremiseClaim_Concat")
        
        premiseClaimLSTM = LSTM(units= embeddingsDimensionalityReduction_FinalDimensions, name= "LSTM")
        
        premiseClaimLSTMEncoding= Bidirectional(layer= premiseClaimLSTM, merge_mode='concat', name= "BiLSTM")(mergedSentenceEncodings)
        
        firstHiddenLayer = Dense(
            units= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            activation= layerActivation, 
            input_dim= embeddingsDimensionalityReduction_FinalDimensions * 2, 
            name= "Hidden1")(premiseClaimLSTMEncoding)
        
        firstDropout= Dropout(dropoutRate, name= "Hidden1Drop")(firstHiddenLayer)
        
        finalNormalization= BatchNormalization(name= "NormLayer")(firstDropout)
        
        softmaxLayer= Dense(units=numberOfClasses, activation='softmax', name= "Softmax")(finalNormalization)
        
        sentenceEncodingModel = Model(inputs=[premiseInput, claimInput], outputs=softmaxLayer)
        
        # Configuring neural net learning process
        sentenceEncodingModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        sentenceEncodingModel.summary()
        
        return sentenceEncodingModel
        
    else:
        raise Exception("Neural network architecture with name '" + str(architectureName) + "' is not defined @ self.neuralNet()! Please provide a valid architecture.")
    


