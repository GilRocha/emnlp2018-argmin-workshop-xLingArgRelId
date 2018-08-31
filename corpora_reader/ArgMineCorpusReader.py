#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import json


from CorpusReader import CorpusReader

class ArgMineCorpusReader(CorpusReader):
    
    
    
    def getADUContentWithContext(self, articleId, targetADUId, sentenceIndexes, dictADUs):
        
        # Article content
        articleFiletext= codecs.open(filename= self.corpusPath + "/" + str(articleId) + "_gold" + ".txt", mode= "r", encoding= "utf-8")
        
        essayCompleteContent= articleFiletext.read()
        
        articleFiletext.close()
        
        aduT0= essayCompleteContent.find(dictADUs[targetADUId]["content"])
        if aduT0 < 0:
            raise Exception("Couldn't find ADU (id= " + str(targetADUId) + ") on text!")
        aduTf= aduT0 + len(dictADUs[targetADUId]["content"]) + 1 # +1 because end indexes are open (standard in this project)
        
        # assuming that the 'sentenceIndexes' list is ordered we just have to take the last index from the indexes that are lower than ArticleId.t0
        closestSentenceBeginIndex= [sentenceIndex for sentenceIndex in sentenceIndexes if sentenceIndex <= aduT0][-1]
        
        adusBoundaries= [essayCompleteContent.find(v["content"]) + len(v["content"]) for v in dictADUs.values()]
        validAduBoundaries= [aduEndIndex for aduEndIndex in adusBoundaries if (aduEndIndex < aduT0) and (aduEndIndex > closestSentenceBeginIndex)]
        
        if len(validAduBoundaries) > 0:
            contextIndex= max(validAduBoundaries)
        else:
            contextIndex= closestSentenceBeginIndex
        
        return essayCompleteContent[contextIndex:aduTf]
    
    
    def getAnnotationForArticle(self, articleFilename):
        
        articleFile= codecs.open(filename= self.corpusPath + "/" + articleFilename + ".json", mode= "r", encoding= "utf-8")
        
        articleJsonContent= json.loads(articleFile.read().replace('\t', '\\t'), encoding= "utf-8")
        
        articleId= int(articleFilename.split("_")[0])
        
        articleAnnotations= {"T": {}, "R": {}}
        
        nodes= {"TextNodes": {}, "RelationNodes": {}}
        
        
        for node in articleJsonContent["nodes"]:
            if node["type"] == "I":
                # ADU
                nodes["TextNodes"][int(node["nodeID"])]= {"content": node["text"]}
            elif node["type"] == "RA":
                # Relation Node -> Support relation
                nodes["RelationNodes"][int(node["nodeID"])]= {"type": "supports"}
            elif node["type"] == "CA":
                # Relation Node -> Attack relation
                nodes["RelationNodes"][int(node["nodeID"])]= {"type": "attacks"}
            
        
        # TextNodes are T nodes!
        articleAnnotations["T"]= nodes["TextNodes"]
        
        # determine relations from json 
        for edge in articleJsonContent["edges"]:
            if int(edge["fromID"]) in nodes["TextNodes"]:
                # To reconstruct the relations we should start from an Text Node
                
                for edge2 in articleJsonContent["edges"]:
                    if int(edge2["fromID"]) == int(edge["toID"]):
                        if int(edge2["toID"]) in nodes["TextNodes"]:
                            articleAnnotations["R"][str(edge["fromID"]) + "_" + str(edge2["toID"])]= {"type": nodes["RelationNodes"][int(edge["toID"])]["type"], "aduSourceId": int(edge["fromID"]), "aduTargetId": int(edge2["toID"])}
                        else:
                            raise Exception("Sanity check failed! Edge2 should connect to a Text Node. Something wrong in the annotation or an unexpected situation was found!")
        
        articleFile.close()
        
        return {"articleId": articleId, "annotations": articleAnnotations}
    
    def getRelationsForArticle(self, articleId, annotationsDict, includeStanceAnnotations= True, includeContext= False):
        
        relations= []
        
        sentenceIndexes= self.determineSentenceIndexes(str(articleId) + "_gold", textLanguage= "portuguese")
        
        for aduSourceId in annotationsDict["annotations"]["T"].keys():
            for aduTargetId in annotationsDict["annotations"]["T"].keys():
                
                aduSourceIdPositiveRelationsDict= {}
                for relation in annotationsDict["annotations"]["R"].values():
                    if int(relation["aduSourceId"]) == aduSourceId:
                        aduSourceIdPositiveRelationsDict[int(relation["aduTargetId"])]= relation["type"]
                
                if aduSourceId != aduTargetId:
                    relationType= "none"
                    if aduTargetId in aduSourceIdPositiveRelationsDict:
                        relationType= unicode(aduSourceIdPositiveRelationsDict[aduTargetId])
                    
                    sourceTextContent= ""
                    if includeContext:
                        sourceTextContent= self.getADUContentWithContext(articleId, aduSourceId, sentenceIndexes, annotationsDict["annotations"]["T"])
                    else:
                        sourceTextContent= annotationsDict["annotations"]["T"][aduSourceId]["content"]
                    
                    targetTextContent= ""
                    if includeContext:
                        targetTextContent= self.getADUContentWithContext(articleId, aduTargetId, sentenceIndexes, annotationsDict["annotations"]["T"])
                    else:
                        targetTextContent= annotationsDict["annotations"]["T"][aduTargetId]["content"]
                    
                    relations.append([unicode(articleId), sourceTextContent, targetTextContent, relationType])
        
        return relations
    
    


argmineReader= ArgMineCorpusReader(corpusPath= os.path.abspath("../data/corpora/pt/"), generatedDatasetPath= os.path.abspath("../data/generatedDatasets/pt/"))

argmineReader.createDatasetForArgumentativeRelationIdentification(".json", "ArgMineCorpus_pt", "ArgMineCorpus_context_pt")
