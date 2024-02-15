#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
import numpy as np
from nltk.corpus import brown
import math
import nltk


CONST_PHI = 0.2
CONST_BETA = 0.45
CONST_ALPHA = 0.2
CONST_PHI = 0.2
CONST_DELTA = 0.875
CONST_ETA = 0.4
total_words = 0
word_freq_brown = {}


def proper_synset(word_one, word_two):
    pair = (None, None)
    maximum_similarity = -1
    synsets_one = wn.synsets(word_one)
    synsets_two = wn.synsets(word_two)
    print("first word :", word_one)
    print("second word", word_two)
    print(synsets_one)
    print(synsets_two)
    if (len(synsets_one) != 0 and len(synsets_two) != 0):
        for synset_one in synsets_one:
            for synset_two in synsets_two:
                similarity = wn.path_similarity(synset_one, synset_two)
                if (similarity == None):
                    sim = -2
                elif (similarity > maximum_similarity):
                    maximum_similarity = similarity
                    pair = synset_one, synset_two
    else:
        pair = (None, None)
    return pair


def length_between_words(synset_one, synset_two):
    length = 100000000
    if synset_one is None or synset_two is None:
        return 0
    elif (synset_one == synset_two):
        length = 0
    else:
        words_synet1 = set([word.name() for word in synset_one.lemmas()])
        words_synet2 = set([word.name() for word in synset_two.lemmas()])
        if (len(words_synet1) + len(words_synet2) > len(words_synet1.union(words_synet2))):
            length = 0
        else:
            length = synset_one.shortest_path_distance(synset_two)
            if (length is None):
                return 0
    return math.exp(-1 * CONST_ALPHA * length)


def depth_common_subsumer(synset_one, synset_two):
    height = 100000000
    if synset_one is None or synset_two is None:
        return 0
    elif synset_one == synset_two:
        height = max([hypernym[1] for hypernym in synset_one.hypernym_distances()])
    else:
        
        hypernym_one = {hypernym_word[0]: hypernym_word[1] for hypernym_word in synset_one.hypernym_distances()}
        hypernym_two = {hypernym_word[0]: hypernym_word[1] for hypernym_word in synset_two.hypernym_distances()}
        common_subsumer = set(hypernym_one.keys()).intersection(set(hypernym_two.keys()))
        if (len(common_subsumer) == 0):
            height = 0
        else:
            height = 0
            for cs in common_subsumer:
                val = [hypernym_word[1] for hypernym_word in cs.hypernym_distances()]
                val = max(val)
                if val > height: height = val

   
    return (math.exp(CONST_BETA * height) - math.exp(-CONST_BETA * height)) / (
                math.exp(CONST_BETA * height) + math.exp(-CONST_BETA * height))


def word_similarity(word1, word2):

    synset_wordone, synset_wordtwo = proper_synset(word1,word2) 
    return length_between_words(synset_wordone, synset_wordtwo) * depth_common_subsumer(synset_wordone, synset_wordtwo)


def I(search_word):
    global total_words
    if (total_words == 0):
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if word not in word_freq_brown:
                    word_freq_brown[word] = 0
                word_freq_brown[word] += 1
                total_words += 1
    count = 0 if search_word not in word_freq_brown else word_freq_brown[search_word]
    ret = 1.0 - (math.log(count + 1) / math.log(total_words + 1))
    return ret


def most_similar_word(word, sentence):
    most_similarity = 0
    most_similar_word = ''
    for w in sentence:
        sim = word_similarity(w, word)
        if sim > most_similarity:
            most_similarity = sim
            most_similar_word = w
    if most_similarity <= CONST_PHI:
        most_similarity = 0
    return most_similar_word, most_similarity


def gen_sem_vec(sentence, joint_word_set):
    semantic_vector = np.zeros(len(joint_word_set))
   
    i = 0
    for joint_word in joint_word_set:
        sim_word = joint_word 
        beta_sim_measure = 1
        if (joint_word in sentence):
            pass
        else:
            sim_word, beta_sim_measure = most_similar_word(joint_word,
                                                           sentence) 
            beta_sim_measure = 0 if beta_sim_measure <= CONST_PHI else beta_sim_measure
        sim_measure = beta_sim_measure * I(joint_word) * I(sim_word)
       
        semantic_vector[i] = sim_measure
        i += 1
    return semantic_vector


def sent_sim(sent_set_one, sent_set_two, joint_word_set):

    print("Sent set one:", sent_set_one)
    print("Sent set two:", sent_set_two)
    sem_vec_one = gen_sem_vec(sent_set_one, joint_word_set)
    sem_vec_two = gen_sem_vec(sent_set_two, joint_word_set)
    print("Semantic vector 1:", sem_vec_one)
    print("Semantic vector 2:", sem_vec_two)
    return np.dot(sem_vec_one, sem_vec_two.T) / (np.linalg.norm(sem_vec_one) * np.linalg.norm(sem_vec_two))


def word_order_similarity(sentence_one, sentence_two):
    print("Sentence one :", sentence_one)
    token_one = word_tokenize(sentence_one)
    print("Sentence two : ", sentence_two)
    token_two = word_tokenize(sentence_two)
    joint_word_set = list(set(token_one).union(set(token_two)))
    r1 = np.zeros(len(joint_word_set))
    r2 = np.zeros(len(joint_word_set))
    en_joint_one = {x[1]: x[0] for x in enumerate(token_one)}
    en_joint_two = {x[1]: x[0] for x in enumerate(token_two)}
    set_token_one = set(token_one)
    set_token_two = set(token_two)
    i = 0
    j=0
    for word in joint_word_set:
        if word in set_token_one:
            r1[i] = en_joint_one[word]  
        else:
    
            sim_word, sim = most_similar_word(word, list(set_token_one))
            if sim > CONST_ETA:
                r1[i] = en_joint_one[sim_word]
            else:
                r1[i] = 0
        i += 1
    for word in joint_word_set:
        if word in set_token_two:
            r2[j] = en_joint_two[word]
        else:
            sim_word, sim = most_similar_word(word, list(set_token_two))
            if sim > CONST_ETA:
                r2[j] = en_joint_two[sim_word]
            else:
                r2[j] = 0
        j += 1
    value = 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))
    print("Word Order Similarity is:", value)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))


def main(sentence_one, sentence_two):
    sent_set_one = set(filter(lambda x: not (x == '.' or x == '?'), word_tokenize(sentence_one)))
    sent_set_two = set(filter(lambda x: not (x == '.' or x == '?'), word_tokenize(sentence_two)))
    joint_word_set = list(sent_set_one.union(sent_set_two))
    sentence_similarity = (CONST_DELTA * sent_sim(sent_set_one, sent_set_two, list(joint_word_set))) + (
                (1.0 - CONST_DELTA) * word_order_similarity(sentence_one, sentence_two))
    return sentence_similarity


def file_sem(f):
    contents = open(f).read().strip()
    ind_sentences = sent_tokenize(contents)
    no_of_sentences = len(ind_sentences)
    sent_sim_matr = np.zeros((no_of_sentences, no_of_sentences))
    i = 0
    print(ind_sentences)
    while (i < no_of_sentences):
        j = i
        while (j < no_of_sentences):
            sent_sim_matr[i][j] = main(ind_sentences[i], ind_sentences[j])
            sent_sim_matr[j][i] = sent_sim_matr[i][j]
            j += 1
        i += 1
    return sent_sim_matr


def intro():
    print("\nEnter a valid option:\n")
    print("1.Sentence Similarity between two files containing different sentences.")
    print("2.Sentence similarity between two sentences\n")
    option = int(input("Your choice : "))
    if option == 1:
        file_one = input("Enter the path of the file :")
        file_two = input("Enter the path of the second file")
        with open(file_one, 'r') as content_file:
            sent_one = content_file.read()
        with open(file_two, 'r') as content_file:
            sent_two = content_file.read()
            prob_sim_sent = main(sent_one, sent_two)
            print("Similarity between the sentences between the 2 file is : \n")
            print(prob_sim_sent)

    
        f_n = file_one[0:len(file_one) - 4:] + "_matrix.txt"
        output_file = open(f_n, 'w')
        output_file.write(str(prob_sim_sent))
    elif option == 2:
        sent_one = input("Enter the first sentence : ")
        sent_two = input("Enter the second sentence two :")
        prob_sim_sent = main(sent_one, sent_two)
        print(prob_sim_sent)
        print("Similarity between "+sent_one+" "+sent_two+" is : ",prob_sim_sent)
    else:
        global max_count
        if max_count < 3:
            print("Wrong Choice Try again"); max_count += 1
        else:
            print("Wrong choice time exceeded!");exit()
        intro()


if __name__ == "__main__":
    print("-------------------Sentence Similarity--------------------------")
    intro()
    print("Want to try once again? if yes press 1 or else 0")
    excited = int(input())
    while (excited == 1):
        intro()
        print("Want to try once again?")
        excited = int(input())


# In[ ]:




