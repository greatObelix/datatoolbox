# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:05:44 2016

@author: mh636c
"""

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim


def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    
    # remove stopwords like 'of' 'or' 'the' 
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    # remove punctuation
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # lemmatization remove inflectional endings only and to return the base or dictionary form of a word,
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return normalized
    
def topic_modeling():
    doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
    doc2 = "My father spends a lot of time driving my sister around to dance practice."
    doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
    doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
    doc5 = "Health experts say that Sugar is not good for your lifestyle."

    # compile documents
    doc_complete = [doc1, doc2, doc3, doc4, doc5]
    doc_clean = [clean(doc).split() for doc in doc_complete] 
    
    # Prepare Document Term Matrix
    # matrix shows a corpus of N documents D1, D2, D3 … Dn and vocabulary size of M words W1,W2 .. Wm.
    # The value of i,j cell gives the frequency count of word Wj in Document Di
    dictionary = gensim.corpora.Dictionary(doc_clean)
    corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
    
    # Running Latent Dirichlet Allocation (LDA) Model
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(corpus, num_topics=4, id2word = dictionary, passes=50)
    
    # Results
    #print(ldamodel.print_topics(num_topics=4, num_words=4)) 
    
 
    
    newdoc = [clean('pastry has high sugar content and fun').split()]    
    newcorpus = [dictionary.doc2bow(doc) for doc in newdoc]
    # get probablity for which topic new/unseen doc fall into
    newtopic = ldamodel.get_document_topics(newcorpus)
    print(newtopic[0])
    # list all topics
    print(ldamodel.show_topics())
    print('\n')
    
    
    return
    
if __name__ == '__main__':
    #import nltk
    #nltk.set_proxy('http://autoproxy.sbc.com/autoproxy.cgi','mh636c','crazy_h0r53')
    #nltk.download()
    topic_modeling()