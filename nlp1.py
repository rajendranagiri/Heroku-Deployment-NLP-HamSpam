# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:23:42 2020

@author: hp
"""
#pip install bert

import bert

dir(bert)

#pip install albert
import albert
dir(albert)

import nltk
dir(nltk)


import spacy 
from spacy.gold import GoldParse 
from spacy.language import EntityRecognizer

import keras  

nlp = spacy.load('en', entity = False, parser = False) 
  
doc_list = [] 
doc = nlp('Llamas make great pets.') 
doc_list.append(doc) 
gold_list = [] 
gold_list.append(GoldParse(doc, [u'ANIMAL', u'O', u'O', u'O'])) 
  
ner = EntityRecognizer(nlp.vocab, entity_types = ['ANIMAL']) 
ner.update(doc_list, gold_list) 
    
#pip install coreNLP
import corenlp
dir(corenlp)

#pip install TextBlob
from textblob import TextBlob
dir(TextBlob)
opinion = TextBlob("EliteDataScience.com is dope.")
opinion.sentiment

python -m TextBlob.download_corpora

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
opinion = TextBlob("EliteDataScience.com is dope!", analyzer=NaiveBayesAnalyzer())
opinion.sentiment

pip install stanza
dir(stanza)
import stanza
