#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''This class extracts features from job descriptions. It is a helper script for the ModelContainer class'''

__author__ = 'Sahil Saxena'
__email__ = 'sahil.saxena@outlook.com'


import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tag import PerceptronTagger
from nltk.data import find
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class Feature_Extractor_and_Processor:

    def __init__(self, job_title_col, url_col, description_col,                  label_col, word_col, encoded_job_title_col, indeed_file, words_file):

        #intialize attributes related to dataset
        self.job_title_col = job_title_col
        self.url_col = url_col
        self.description_col = description_col
        self.label_col = label_col
        self.word_col = word_col
        self.encoded_job_title_col = encoded_job_title_col
        
        #load the scraped files
        self.df_indeed = self._load_data(indeed_file)
        self.df_words = self._load_data(words_file)
                
        #initialize attributes related to extracted features
        self.job_description = None
        self.word_list = None
        self.features_list_single = []
        self.features_list_phrase = []
        self.topk_single = None
        self.topk_phrase = None
        self.topk_full = None
        self.df_single = pd.DataFrame()
        self.df_phrase = pd.DataFrame()
        self.df = pd.DataFrame()
        self.df_tools = pd.DataFrame()
        self.df = pd.DataFrame()
        self.top_tools_dict = {}
        
        # Initialize phrase tree
        self.grammar = self._initialize_grammar()
        self.stop = self._initialize_stopwords()
        self.text = """ initialize """
        self.tagger = PerceptronTagger()
        self.pos_tag = self.tagger.tag
        self.chunker = nltk.RegexpParser(self.grammar)
        self.taggedToks = self.pos_tag(re.findall(r'\w+', self.text))
        self.tree = self.chunker.parse(self.taggedToks)
        
        #perform pre-processing, feature extraction and post-processing
        self._execute_pre_processing()
        self._execute_feature_extraction()
        self._execute_post_processing()
        
    
    def _load_data(self, file):
        df_temp = pd.read_csv(file, index_col=[0])
        return df_temp
    
    def _join_lists(self, lst1, lst2):
        lst_full = lst1 + lst2
        return lst_full
    
    def _return_lists_intersection(self, lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2] 
        return lst3
        
    def _merge_dfs_horizontally(self, df1, df2, common_cols):
        '''Efficient joining of two dataframes horizontally knowing the index of columns that are common between them'''
        df_temp = pd.concat([df1, df2.iloc[:, common_cols:]], axis=1)
        df_temp.drop_duplicates(inplace = True)
        df_temp.reset_index(drop = True, inplace = True)
        return df_temp
    
    def _job_title_encode(self, x):
        if x == "Data Scientist":
            return 0
        elif x == "Data Analyst":
            return 1
        elif x == "Machine Learning Engineer":
            return 2
        elif x == "Data Engineer":
            return 3
    
    def _initialize_grammar(self):    
        '''This grammar is described in the paper by S. N. Kim,
        # T. Baldwin, and M.-Y. Kan.
        # Evaluating n-gram based evaluation metrics for automatic
        # keyphrase extraction.
        # Technical report, University of Melbourne, Melbourne 2010.''' 

        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        return grammar
    
    
    def _initialize_stopwords(self):    
        '''set stop words for the processing steps''' 
        stop = set(stopwords.words('english'))
        stop.add('achieve'); stop.add('analysis'); stop.add('analyst'); 
        stop.add('b'); stop.add('based'); stop.add('basic'); stop.add('casual'); stop.add('class'); 
        stop.add('covid'); stop.add('delivering'); stop.add('due'); stop.add('e'); stop.add('ec'); 
        stop.add('engineer'); stop.add('engineering'); stop.add('es'); stop.add('fluent'); 
        stop.add('full'); stop.add('gap'); stop.add('goal'); stop.add('great'); stop.add('health'); stop.add('hope');
        stop.add('hour'); stop.add('j'); stop.add('lean'); stop.add('learning'); stop.add('machine learning'); 
        stop.add('machine'); stop.add('math'); stop.add('p'); stop.add('pilot'); stop.add('pizza'); 
        stop.add('processing'); stop.add('remotely'); stop.add('required'); stop.add('requirements'); stop.add('science'); 
        stop.add('scientist'); stop.add('scientists'); stop.add('skills'); stop.add('source'); stop.add('spin'); 
        stop.add('state'); stop.add('strong'); stop.add('temporarily'); stop.add('types'); stop.add('word'); 
        stop.add('working'); stop.add('world'); stop.add('year'); stop.add('plus'); stop.add('inform'); 
        stop.add('cool'); stop.add('offer'); stop.add('ability'); stop.add('get'); stop.add('self'); 
        stop.add('provide'); stop.add('within'); stop.add('years'); stop.add('experience'); stop.add('performance'); 
        stop.add('go'); stop.add('training'); stop.add('flex'); stop.add('k'); stop.add('f'); stop.add('ensure');
        stop.add('responsible')
        return stop
    
    def _clean_words(self, x):
        '''basic cleaning of text'''
        x = x.lower()
        x = re.sub(r'\([^)]*\)', "", x)
        x = re.sub("_[\_(\_[].*?[\)\]]", "", x)
        x = x.replace("_", " ")
        return x
    
    def _clean_description(self, x):
        letters_only = re.sub("[^a-zA-Z]",  # Search for all non-letters
                              " ",          # Replace all non-letters with spaces
                              str(x))
        return letters_only
    
    def _execute_pre_processing(self):
        self._job_description_pre_processing(self.description_col, self.url_col)
        self._words_pre_processing(self.word_col)
        
    def _job_description_pre_processing(self, description_col, url_col):
        self.df_indeed[description_col] = self.df_indeed[description_col].apply(lambda x: self._clean_description(str(x)))
        self.df_indeed.drop_duplicates(subset = [url_col], inplace = True)
        self.df_indeed.reset_index(drop = True, inplace = True)
        self.job_description = self.df_indeed[description_col].values
        
    def _words_pre_processing(self, word_col): 
        self.df_words[word_col] = self.df_words[word_col].apply(lambda x: self._clean_words(str(x)))
        self.df_words.drop_duplicates(subset = [word_col], inplace = True)
        self.df_words.reset_index(drop = True, inplace = True)
        self.word_list = self.df_words[word_col].tolist()

    def _save_data(self, df, file):
        df.to_csv(file)
        
    def _do_subset(self, counter, feature_names):
        return Counter({k: counter.get(k, 0) for k in feature_names})    

    # generator, generate leaves one by one
    def _leaves(self, tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP' or t.label()=='JJ' or t.label()=='RB'):
            yield subtree.leaves()
 
    def _normalise(self, word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        # ignore stemming and lemmatization
        # word = stemmer.stem(word)
        # word = lemmatizer.lemmatize(word)
        return word

    # stop-words and length control
    def _acceptable_word(self, word):
        """Checks conditions for acceptable word: length, stopword."""
        accepted = bool(1 <= len(word) <= 40
            and word.lower() not in self.stop)
        return accepted

    # generator, create item once a time
    def _get_terms(self, tree):
        for leaf in self._leaves(tree):
            term = [self._normalise(w) for w,t in leaf if self._acceptable_word(w) ]
            # Phrase only
            if len(term)>1:
                yield term
    
    # Flatten phrase lists to get tokens for analysis
    def _flatten(self, npTokenList):
        finalList =[]
        for phrase in npTokenList:
            token = ''
            for word in phrase:
                    token += word + ' '
            finalList.append(token.rstrip())
        return finalList  

    def preProcess(self, text):
        text = text.lower()
        return text
    
    
    def get_subset_counter_list(self, job_description_values, features_list_single, features_list_phrase):
        #make a counter list of all words and bi-words
        #then only take subset of it i.e. the ones that appear as features in the full_df
        counter_single = Counter()
        counter_phrase = Counter()

        #make counter for all single words
        for review in job_description_values:
            counter_single.update([word.lower() 
                            for word 
                            in re.findall(r'\w+', review) 
                            if word.lower() not in self.stop and len(word) > 0])

        #make counter for all phrases
        for review in job_description_values:
            counter_phrase.update(self._flatten([word
                            for word 
                            in self._get_terms(self.chunker.parse(self.pos_tag(re.findall(r'\w+', review)))) 
                            ]))

        #take subset as per feature lists, return a counter object
        counter_subset_single = self._do_subset(counter_single, features_list_single)
        counter_subset_phrase = self._do_subset(counter_phrase, features_list_phrase)

        #remove all zero count entries
        counter_subset_single = Counter({k: c for k, c in counter_subset_single.items() if c > 0})
        counter_subset_phrase = Counter({k: c for k, c in counter_subset_phrase.items() if c > 0})

        #convert into list of tuples
        topk_single = list(counter_subset_single.items())
        topk_phrase = list(counter_subset_phrase.items())
        return topk_single, topk_phrase

    
    def _boolean_encoding_single(self, k=50):
        '''Boolean Encode Single Terms '''
        
        #if no topk is passed, make a topk tuple list of frequent terms
        if self.topk_single == None:
          # Top-k frequent terms
            counter = Counter()
            for review in self.job_description:
                counter.update([word.lower() 
                              for word 
                              in re.findall(r'\w+', review) 
                              if word.lower() not in self.stop and len(word) > 0])
            self.topk_single = counter.most_common(k)      

        #Find out if a particular review has the word from topk list
        freqReview = []
        for i in range(len(self.job_description)):
            # you feed a list of elements in counter ["a", "a", "c", "r"] then counter returns counter object {"a":2, "c":1, "r":1}
            tempCounter = Counter([word.lower() for word in re.findall(r'\w+',self.job_description[i])])
            topkinReview = [1 if tempCounter[word] > 0 else 0 for (word,wordCount) in self.topk_single]
            freqReview.append(topkinReview)

        #Prepare freqReviewDf
        freqReviewDf = pd.DataFrame(freqReview)
        dfName = []
        for c in self.topk_single:
            dfName.append(c[0])
        freqReviewDf.columns = dfName
        self.df_single = self.df_indeed.join(freqReviewDf)
        self.df_single.drop_duplicates(inplace = True)
        self.df_single.reset_index(drop = True, inplace = True)
        

    def _boolean_encoding_phrase(self, k=50):
        '''Boolean Encode Phrases '''

        # #if no topk is passed, make a topk tuple list of frequent terms
        if self.topk_phrase == None:
            counter = Counter()
            for review in self.job_description:
                  counter.update(self._flatten([word
                                  for word 
                                  in self._get_terms(self.chunker.parse(self.pos_tag(re.findall(r'\w+', review)))) 
                                  ]))
            self.topk_phrase = counter.most_common(k)        

        #Find out if a particular review has the word from topk list
        freqReview = []
        for i in range(len(self.job_description)):
            tempCounter = Counter(self._flatten([word 
                                           for word 
                                           in self._get_terms\
                                                 (self.chunker.parse(self.pos_tag(re.findall(r'\w+',self.job_description[i]))))]))
            topkinReview = [1 if tempCounter[word] > 0 else 0 for (word,wordCount) in self.topk_phrase]
            freqReview.append(topkinReview)

        #Prepare freqReviewDf
        freqReviewDf = pd.DataFrame(freqReview)
        dfName = []
        for c in self.topk_phrase:
            dfName.append(c[0])
        freqReviewDf.columns = dfName
        self.df_phrase = self.df_indeed.join(freqReviewDf)
        self.df_phrase.drop_duplicates(inplace = True)
        self.df_phrase.reset_index(drop = True, inplace = True)
    
    def _execute_feature_extraction(self, max_df = 0.9, min_df = 0.2):

        #tf-idf matrix for single words
        tfidf_single = TfidfVectorizer(stop_words = self.stop, max_df = max_df, min_df = min_df,                                        ngram_range = (1,1))
        tfs_single = tfidf_single.fit_transform(self.job_description.astype('U'))
        self.features_list_single = tfidf_single.get_feature_names()

        #tf-idf matrix for phrase
        tfidf_phrase = TfidfVectorizer(stop_words = self.stop, max_df = max_df, ngram_range = (2,2),                                        min_df = min_df)
        tfs_phrase = tfidf_phrase.fit_transform(self.job_description.astype('U'))
        self.features_list_phrase = tfidf_phrase.get_feature_names()

        #build dataframe
        corpus_index = range(1, len(self.job_description)+1)
        self.df_single = pd.DataFrame(tfs_single.todense(), index=corpus_index, columns= self.features_list_single)
        self.df_phrase = pd.DataFrame(tfs_phrase.todense(), index=corpus_index, columns= self.features_list_phrase)
        
        #add technology words to feature lists
        self.features_list_single = self.word_list + self.features_list_single
        self.features_list_phrase = self.word_list + self.features_list_phrase
        self.features_list_single = list(dict.fromkeys(self.features_list_single))
        self.features_list_phrase = list(dict.fromkeys(self.features_list_phrase))
        
        #get the topkwords tuple list for all the terms in the feature lists
        self.topk_single, self.topk_phrase = self.get_subset_counter_list(self.job_description,                                                                             self.features_list_single, self.features_list_phrase)
        
        #generate a boolean encoded dataframe of all features for single and phrase lists
        self._boolean_encoding_single()
        self._boolean_encoding_phrase()
        
    def _execute_post_processing(self):
        
        #merge dataframes and topk
        self.df = self._merge_dfs_horizontally(self.df_single, self.df_phrase, common_cols = 4)
        self.topk_full = self._join_lists(self.topk_single, self.topk_phrase)
        
        #add encoding for job_title
        self.df[self.encoded_job_title_col] = self.df[self.label_col].apply(lambda x: self._job_title_encode(x))
        
        #initialize the df_tools dataframe which is used for the models
        common_columns_list = self._return_lists_intersection(self.df.columns.tolist(), self.word_list)
        general_columns = [self.job_title_col, self.url_col, self.description_col, self.label_col, self.encoded_job_title_col]
        filter_cols = general_columns + common_columns_list
        self.df_tools = self.df[filter_cols]
        
        #initiate top_tools_dict which stores the top technical terms used in job postings
        top_tools_dict = {}
        first_col_index_with_features = 5
        take_top = 10

        for title in self.df_tools[self.label_col].value_counts().index.tolist():

            df_tools_ds = self.df_tools[self.df_tools[self.label_col] == title]
            df_tools_ds = df_tools_ds.iloc[:, first_col_index_with_features:]
            term_frequency_series = df_tools_ds.sum(axis = 0).sort_values(ascending = False)
            df_term_frequency = pd.DataFrame(term_frequency_series)
            df_term_frequency.reset_index(inplace = True, drop = False)
            df_term_frequency.columns = ["tool", "count"]
            self.top_tools_dict[title] = df_term_frequency["tool"][0:take_top+1].tolist()
              
    

