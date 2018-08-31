#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs

from CorpusReader import CorpusReader

class ArgumentativeEssayCorpusReader(CorpusReader):
    
    def determineEssayParagraphsIndexes(self, essayFilename):
        
        paragraphsIndexes= []
        
        essayFiletext= codecs.open(filename= self.corpusPath + "/" + essayFilename + ".txt", mode= "r", encoding= "utf-8")
        
        essayCompleteContent= essayFiletext.read()
        
        splittedContentByEndLine= essayCompleteContent.split("\n")
        
        #charIndexCounter= 0
        
        # first two new lines are not paragraph delimiters but should be included in the counter
        # Note: one's below correspond to the newline character
        charIndexCounter = len(splittedContentByEndLine[0]) + 1 + len(splittedContentByEndLine[1]) + 1
        
        # NOTE: index of the paragraph indicating the beginning of the essay body is included as the first paragraph index
        paragraphsIndexes.append(charIndexCounter - 1)
        
        for paragraph in splittedContentByEndLine[2:]:
            charIndexCounter= charIndexCounter + len(paragraph) + 1
            paragraphsIndexes.append(charIndexCounter - 1)
        
        return paragraphsIndexes
    
    def getADUContentWithContext(self, articleId, targetADUId, sentenceIndexes, dictADUs):
        
        # Article content
        articleFiletext= codecs.open(filename= self.corpusPath + "/" + str(articleId) + ".txt", mode= "r", encoding= "utf-8")
        
        essayCompleteContent= articleFiletext.read()
        
        articleFiletext.close()
        
        # assuming that the 'sentenceIndexes' list is ordered we just have to take the last index from the indexes that are lower than ArticleId.t0
        closestSentenceBeginIndex= max([sentenceIndex for sentenceIndex in sentenceIndexes if sentenceIndex <= int(dictADUs[targetADUId]["t0"])])
        
        adusBoundaries= [int(v["tf"]) for v in dictADUs.values()]
        validAduBoundaries= [aduEndIndex for aduEndIndex in adusBoundaries if (aduEndIndex < int(dictADUs[targetADUId]["t0"])) and (aduEndIndex > closestSentenceBeginIndex)]
        
        if len(validAduBoundaries) > 0:
            contextIndex= max(validAduBoundaries)
        else:
            contextIndex= closestSentenceBeginIndex
        
        return essayCompleteContent[contextIndex:int(dictADUs[targetADUId]["tf"])]
    
    def getAnnotationForArticle(self, articleFilename):
        
        currentEssay= codecs.open(filename= self.corpusPath + "/" + articleFilename + ".ann", mode= "r", encoding= "utf-8")
        
        essayId= articleFilename
        
        #essayDict= {"essayId": essayId, "annotations": {"T": [], "R": [], "A": []}}
        essayAnnotationsContent= {"T": {}, "R": {}, "A": {}}
        
        for line in currentEssay:
            
            # separated by tabs
            splittedLineByTabs= line.split("\t")
            
            
            if splittedLineByTabs[0][0] == "T":
                # 'T' (i.e. ADU) line
                aduId= splittedLineByTabs[0]
                
                aduInfo= splittedLineByTabs[1].split(" ")
                aduType= aduInfo[0]
                aduInitialCharIndex= aduInfo[1]
                aduEndCharIndex= aduInfo[2]
                
                aduContent= splittedLineByTabs[2].rstrip()
                
                essayAnnotationsContent["T"][aduId] = {"type": aduType, "t0": int(aduInitialCharIndex), "tf": int(aduEndCharIndex), "content": aduContent}
                
            elif splittedLineByTabs[0][0] == "R":
                # 'R' (i.e. Relation) line
                relationId= splittedLineByTabs[0]
                
                relationInfo= splittedLineByTabs[1].split(" ")
                relationType= relationInfo[0]
                relationSourceId= relationInfo[1].split(":")[1]
                relationTargetId= relationInfo[2].split(":")[1]
                
                essayAnnotationsContent["R"][relationId] = {"type": relationType, "aduSourceId": relationSourceId, "aduTargetId": relationTargetId}
                
            elif splittedLineByTabs[0][0] == "A":
                # 'A' (i.e. stance) line
                # Note: considered as a special 'R' line, containing a relation to each of the major claims
                stanceId= splittedLineByTabs[0]
                
                stanceInfo= splittedLineByTabs[1].split(" ")
                stanceSourceId= stanceInfo[1]
                stanceType= stanceInfo[2].rstrip()
                
                essayAnnotationsContent["A"][stanceId] = {"type": stanceType, "aduSourceId": stanceSourceId}
        
        currentEssay.close()
        
        return {"articleId": essayId, "annotations": essayAnnotationsContent}
    
    def getPositiveRelationsFromAnnotationsDict(self, essayId, annotationsDict, includeStanceAnnotations= False):
        
        # list of relations, where each element is structured in the following format: "essayId", "source", "target", "relationType"
        relations = []
        
        # 'R' annotations
        for relation in annotationsDict["R"].values():
            relations.append([essayId, annotationsDict["T"][relation["aduSourceId"]]["content"], annotationsDict["T"][relation["aduTargetId"]]["content"], relation["type"]])
        
        if includeStanceAnnotations:
            # 'A' annotations
            for relation in annotationsDict["A"].values():
                for majorClaimId in [aduId for aduId, aduInfo in annotationsDict["T"].items() if aduInfo["type"] == "MajorClaim"]:
                    if relation["type"] == "For":
                        relationType= "supports"
                    else:
                        relationType= "attacks"
                    relations.append([essayId, annotationsDict["T"][relation["aduSourceId"]]["content"], annotationsDict["T"][majorClaimId]["content"], relationType])
            
        
        return relations
    
    
    def getRelationsForArticle(self, articleId, annotationsDict, includeStanceAnnotations= True, includeContext= False):
        
        # list of relations, where each element is structured in the following format: "articleId", "source", "target", "relationType"
        relations= []
        
        paragraphIndexes= self.determineEssayParagraphsIndexes(articleId)
        
        sentenceIndexes= self.determineSentenceIndexes(articleId, textLanguage= "english")
        
        for paragraphId in range(len(paragraphIndexes) - 1):
            adusInParagraph= self.getADUsInParagraph(annotationsDict["annotations"], paragraphIndexes[paragraphId], paragraphIndexes[paragraphId + 1], includeStanceAnnotations= includeStanceAnnotations)
            
            for aduSourceId in adusInParagraph:
                aduSourceIdPositiveRelationsDict= {}
                
                for relationInfo in annotationsDict["annotations"]["R"].values():
                    if relationInfo["aduSourceId"] == aduSourceId:
                        # current relation includes the "aduSourceId" as the source of the relation
                        aduSourceIdPositiveRelationsDict[relationInfo["aduTargetId"]]= relationInfo["type"]
                
                if includeStanceAnnotations:
                    for relationInfo in annotationsDict["annotations"]["A"].values():
                        if relationInfo["aduSourceId"] == aduSourceId:
                            for majorClaimId in [aduId for aduId, aduInfo in annotationsDict["annotations"]["T"].items() if aduInfo["type"] == "MajorClaim"]:
                                if majorClaimId in adusInParagraph:
                                    # just make a relations if corresponding MajorClaim is in the same paragraph
                                    if relationInfo["type"] == "For":
                                        relationType= "supports"
                                    else:
                                        relationType= "attacks"
                                    
                                    aduSourceIdPositiveRelationsDict[majorClaimId]= relationType
                
                for aduTargetId in adusInParagraph:
                    
                    if not (aduSourceId == aduTargetId):
                        relationType= "none"
                        if aduTargetId in aduSourceIdPositiveRelationsDict:
                            relationType= aduSourceIdPositiveRelationsDict[aduTargetId]
                        
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
                        
                        relations.append([articleId, sourceTextContent, targetTextContent, relationType])
                    
            
        return relations
    
    # paragraphStartIndex: character index for the first char in the paragraph
    # paragraphEndIndex: character index for the newline that determines the end of the paragraph
    def getADUsInParagraph(self, annotationsDict, paragraphStartIndex, paragraphEndIndex, includeStanceAnnotations= False):
        
        textNodesInParagraph= []
        
        for textNodeId, textNodeInfo in annotationsDict["T"].items():
            if (textNodeInfo["t0"] >= (paragraphStartIndex + 1)) and (textNodeInfo["tf"] <= paragraphEndIndex):
                
                # Text node in paragraph
                if includeStanceAnnotations:
                    # add all ADUs
                    textNodesInParagraph.append(textNodeId)
                elif not (textNodeInfo["type"] == "MajorClaim"):
                    # In this scenario, ignore "MajorClaim"'s
                    textNodesInParagraph.append(textNodeId)
                
        
        return textNodesInParagraph
        
    
    


aecReader= ArgumentativeEssayCorpusReader()

aecReader.createDatasetForArgumentativeRelationIdentification(".ann", "ArgEssaysCorpus_en", "ArgEssaysCorpus_context_en", includeStanceAnnotations= True)
