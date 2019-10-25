'''
1. open the dataset
2. remove stop words, and apply lematization

'''
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from os import listdir
import multiprocessing 
'''
A generic function to process all textual data
'''
loc = 'dataset/'
def GetProcessed(dname):
    lmtzr = WordNetLemmatizer()
    uniq_wrds = set()
    # tokenizes the sentence by considering only alpha numeric characters
    tokenizer = RegexpTokenizer(r'\w+')
    with open(loc+dname,'r') as fp:
        container=''
        for line in tqdm(fp):
            # get the textual contents. This depends on the structure of data
            contens = line.split('\t')
            tokens = tokenizer.tokenize(contens[1])
#             tagged = pos_tag(tokens)
            ''''this function filters out words that are not a 
            part of pos taggs. Does heavy filteration''' 
#             nouns = [word for word,pos in tagged \
#                      if (pos == 'NN' or pos == 'NNP' or pos == 'NNS'\
#                          or pos == 'NNPS' or pos=='JJ')]
            downcased = [x.lower() for x in tokens]
            downcased = [lmtzr.lemmatize(x) for x in downcased]
            downcased = [x for x in downcased if x not in stopwords.words('english')]
            container += contens[0]+'\t'+' '.join(downcased)+'\n'
        return container
#             uniq_wrds.update(downcased)
#             print i
#     nu_fil.write('UniqWords \t'+str(len(uniq_wrds))+'\n')
#     nu_fil.close()
#     print 'uniq words {}'.format(len(uniq_wrds))

def mp_handler():
    mypath = 'dataset/'
    p = multiprocessing.Pool(8)
    filenames = [f for f in listdir(mypath) if f.startswith('Musical_Instruments_reviews_')]
    with open('results.txt', 'w') as f:
        for result in p.imap(GetProcessed, filenames):
            # (filename, count) tuples from worker
            f.write(result)
            
if __name__ == '__main__':
    
    mp_handler()
#     GetProcessed('Musical_Instruments_reviews.txt')

    