# conda install nltk and do nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
# do nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from string import punctuation
from tqdm import tqdm

'''for the following model see: do pip install contractions 
(see https://github.com/kootenpv/contractions) and do pip install textsearch
since, there is no nltk module for contractions.

1. pip install textsearch
2. pip install contractions
3. nltk.download('stopwords')
4. nltk.download('wordnet')'''
import contractions
import numpy as np
import multiprocessing
import os
import logging
from gensim.test.utils import get_tmpfile, datapath
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

'''
A generic function to process all textual data
'''
HOME_DIR = os.path.expanduser('~')+'/Google Drive/DataRepo/'
F_PTH = HOME_DIR+'AmazonReviews/Musical_Instrument/PretrainedEmbeddings/'
FIL_A = HOME_DIR +'PretrainedEmbeddings/Glove/glove.6B.50d.txt'
FIL_B = 'AmazonMusicToGlove-'
glove_file = datapath(FIL_A)
word2vec_glove_file = get_tmpfile('glove.to.word2vec.txt')
glove2word2vec(glove_file, word2vec_glove_file)
WORD_MODEL = KeyedVectors.load_word2vec_format(word2vec_glove_file)

# the method MeanEmbeddingVectorizer is borrowed from https://github.com/TomLin/Playground

def MeanEmbeddingVectorizer(docs):

    # glove_mean_vec_tr = MeanEmbeddingVectorizer(glove_WORD_MODEL)
    # WORD_MODEL = glove_mean_vec_tr.transform(docs)

    vector_size = WORD_MODEL.wv.vector_size
    doc_vecs = []
    for word in docs:
        if word in WORD_MODEL.wv.vocab:
            doc_vecs.append(WORD_MODEL.wv.get_vector(word))

    if not doc_vecs:  # empty words
        # If a text is empty, return a vector of zeros.
        # logging.warning("cannot compute average owing to no vector for {}".format(docs))
        doc_vecs.append(np.zeros(vector_size))

    return np.array(doc_vecs).mean(axis=0)

def __textOperations(text):
    lmtzr = WordNetLemmatizer()
    # start by removing contractions
    text = contractions.fix(text)
    # tokenizes the sentence by considering only alpha numeric characters
    tokens = word_tokenize(text)
    tokens = [x.lower() for x in tokens]
    tokens = [x for x in tokens if x not in punctuation]
    tokens = [lmtzr.lemmatize(x) for x in tokens]
    tokens = [x for x in tokens if x not in stopwords.words('english')]
    tokens = MeanEmbeddingVectorizer(tokens)
    doc_len = len(tokens)
    return tokens
    # return (' '.join(tokens), doc_len)



def PurifyText(df, field):
    # check if processed dataframe already exists, if yes, load and return this dataframe
    fnam = F_PTH + FIL_B + field + '.pkl'
    mx_doc_len = 0
    if os.path.exists(fnam):
        print('loaded existing embedding...')
        df = pd.read_pickle(fnam)
        for index, row in df.iterrows():
            # if isinstance(row[field],str):
            mx_doc_len = max(len(row[field]), mx_doc_len)
        return (df, mx_doc_len)
    # if there are no existing embeddings, then start purifying the text
    cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpus)
    mx_doc_len = 0
    embeded_docs = list(tqdm(p.imap(__textOperations, df[field]), total=len(df)))
    df['Embedings'] = pd.Series(embeded_docs).values
    # # some rows have nan values in description, so drop such rows
    df = df.dropna()
    df.to_pickle(fnam)
    return (df, mx_doc_len)

