import numpy as np
from collections import defaultdict
import pandas as pd
import sys


system_file=open("system_file","w")
f_in=open(sys.argv[1],"r")
lines = (line.rstrip() for line in f_in) # All lines including the blank ones
lines = (line for line in lines if line) # Non-blank lines
list_of_bigrams=list()
tag_of = dict() #key: word ,value: list of tags the word is tagged in in training
count_of = dict() #key: word ,value: count of that word in the training
tag_count_of = dict() #key: tag, value: count of that tag in traing
tag1='start'   
for line in lines:            #making bigrams
    #print (line)
    index,word,word_tag = line.split('\t')
    if word not in tag_of.keys():
        tag_of[word]=list()
        count_of[word]=0
    tag_of[word].append(word_tag)
    count_of[word] +=1
    if (index == '1'):
        tag1 = 'start'
    tag2=word_tag
    list_of_bigrams.append([tag1,tag2])
    tag1 = tag2
    if word_tag in tag_count_of:
        tag_count_of[word_tag] +=1
    else:
        tag_count_of[word_tag] = 1
deleted_words = list()
tags_ = list()
counts_ = 0
#handling 1 freq words as unk
for i in count_of.keys():
    if(count_of[i] == 1):
        deleted_words.append(i)
        counts_ += 1
        tags_ = tags_ + tag_of[i]
count_of['unk'] = counts_
tag_of['unk'] = tags_
for i in deleted_words:    
    del count_of[i]
    del tag_of[i]

list_of_tags = list(tag_count_of.keys())
list_of_tags.append('start')
list_of_words = list(tag_of.keys())



tag_l = len(list_of_tags)
bigram_counts = np.zeros((tag_l,tag_l))
for bigram in list_of_bigrams:
    t1 = list_of_tags.index(bigram[0])
    
    t2 = list_of_tags.index(bigram[1])
    
    bigram_counts[t1][t2] += 1
    

index_of_start = list_of_tags.index('start')
list_of_tags.remove('start')
tag_l -=1
smooth_prob = np.zeros((tag_l,tag_l))
smooth_prob[:]=0.0001 #add k smoothing k=0.0001
initial_prob = np.zeros((tag_l,1))


for i in list_of_tags:
    for j in list_of_tags:
        ii = list_of_tags.index(i)
        jj = list_of_tags.index(j)
        smooth_prob[ii][jj] += bigram_counts[ii][jj]
     
    ii = list_of_tags.index(i)
    smooth_prob[ii] /= smooth_prob[ii].sum()
    initial_prob[ii] = bigram_counts[index_of_start][ii]

test_words = dict()
predicted_tags = dict()
indices=dict()
f_out=open(sys.argv[2],"r")
lines = (line.rstrip() for line in f_out) # All lines including the blank ones
lines = (line for line in lines if line) # Non-blank lines
index = -1
test_words[0]=list()
indices[0]=list()
for i in lines:
    ind,word= i.split('\t')
    #print ("first word is:"+str(ind))
    if (ind == '1'):
        index +=1
        test_words[index] = list()
        indices[index]=list()
    test_words[index].append(word)
    indices[index].append(ind)


for test_key in range(0,index+1):
    t = test_words[test_key]
    
    

    emmision = np.zeros((len(t),tag_l))
    #emmision prob matrix of test sentence
    for word in range(0,len(t)):
        for state in range(0,tag_l):
            if (t[word] in list_of_words):
                emmision[word][state] += tag_of[t[word]].count(list_of_tags[state])
            else:
                
                emmision[word][state] += tag_of['unk'].count(list_of_tags[state])
            emmision[word][state] /= tag_count_of[list_of_tags[state]]
    

    #viterbi
    t1 = defaultdict(dict)
    t2 = defaultdict(dict)
    z = [None]*(len(t))
    X = [None]*(len(t))
    l = -9999
    for state in range(0,tag_l):
        t1[state][0] = initial_prob[state] * emmision[0][state]
        t2[state][0] = 0
    
    for i in range(1,len(t)):
        for j in range(0,tag_l):
            l = -9999
            arg_l = None
            for k in range(0,tag_l):
                if (l < (t1[k][i-1] * smooth_prob[k][j])):
                    l = t1[k][i-1] * smooth_prob[k][j]
                    arg_l = list_of_tags[k]
            t1[j][i] = emmision[i][j] * l
            t2[j][i] = arg_l
            
    
    l = -999
    arg_l = None
    for k in range(0,tag_l):
        if(l < t1[k][len(t)-1]):
            l = t1[k][len(t)-1]
            arg_l = list_of_tags[k]
    
    z[len(t)-1] = arg_l
    
    
    for i in range(len(t)-1,0,-1):
        z[i-1] = t2[list_of_tags.index(z[i])][i]
        
    
    for write_key in range(0,len(z)):
        system_file.write(indices[test_key][write_key])
        system_file.write('\t')
        system_file.write(test_words[test_key][write_key])
        system_file.write('\t')
        system_file.write(z[write_key])
        system_file.write('\n')
    if (test_key == index):
        continue
    else:
        system_file.write('\n')
    emmision =None
    t1 =None
    t2 =None
    z = None
    X = None


