#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''This script handles wordcloud generation, perform clustering and develops network graph'''

__author__ = 'Sahil Saxena'
__email__ = 'sahil.saxena@outlook.com'

import numpy as np
import pandas as pd
import math

from tqdm import tqdm
import math
from scipy.stats import chi2_contingency
from wordcloud import WordCloud

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#import nlp
import re

#import counter
from collections import Counter

#sklearn
from sklearn.cluster import SpectralClustering

#network graph library
from pyvis.network import Network

#import processing class
from _02_FeatureExtractor import Feature_Extractor_and_Processor

class Modeling_and_Visualization:
    
    def __init__(self, job_title_col, url_col, job_description_col,                  label_col, word_col, encoded_job_title_col, indeed_file, words_file, number_words_each_cluster):

        self.fe = Feature_Extractor_and_Processor(job_title_col, url_col, job_description_col, label_col,                                      word_col, encoded_job_title_col, indeed_file, words_file)
    
        self.df_tools_with_clusters = self.fe.df_tools.copy()
        self.number_words_each_cluster = number_words_each_cluster
        self.top_words_by_cluster_dict = {}
        self.topwords_by_title_dict = {}
        self.pmi_dict = {}
        self.df_ds_subset_single = None
        self.df_ds_subset_phrase = None
        self.topk_single = []
        self.topk_phrase = []
        self.net_dict = {}
        self.temp_dict = {}
        self.df_edge_dict = {}
    
    def _load_data(self, file):
        return pd.read_csv(file, index_col=[0])    
    
    
    # Simple example of getting pairwise mutual information of a term
    def _pmiCal(self, df_temp, x, label_col):
        pmilist=[]

        #cluster labeling starts from 0
        number_of_labels = df_temp[label_col].value_counts().shape[0]

        for i in range(0, number_of_labels):
            for j in [0,1]:    
                px = sum(df_temp[label_col]==i)/len(df_temp)
                py = sum(df_temp[x]==j)/len(df_temp)
                pxy = len(df_temp[(df_temp[label_col]==i) & (df_temp[x]==j)])/len(df_temp)
                if pxy==0:#Log 0 cannot happen
                    pmi = math.log((pxy+0.0001)/(px*py))
                else:
                    pmi = math.log(pxy/(px*py))
                pmilist.append([i]+[j]+[px]+[py]+[pxy]+[pmi])
        pmidf = pd.DataFrame(pmilist)
        pmidf.columns = ['x','y','px','py','pxy','pmi']
        return pmidf
    
    
    def _pmiIndivCal(self, df_temp,x,gt, label_col):
        px = sum(df_temp[label_col]==gt)/len(df_temp)
        py = sum(df_temp[x]==1)/len(df_temp)
        pxy = len(df_temp[(df_temp[label_col]==gt) & (df_temp[x]==1)])/len(df_temp)
        if pxy==0:#Log 0 cannot happen
            pmi = math.log((pxy+0.0001)/(px*py))
        else:
            pmi = math.log(pxy/(px*py))
        return pmi
    
       
    # Compute PMI for all terms and all possible labels
    def _pmiForAllCal(self, df_temp, topk, label_col):
        '''Calculate pmi for top k and store them into one pmidf dataframe '''

        pmilist = []
        pmi_label_dict = {}

        #initiate a dictionary of empty lists 
        for label in df_temp[label_col].value_counts().index.tolist():
            pmi_label_dict[label] = []

        for word in tqdm(topk):
            pmilist.append([word[0]]+[self._pmiCal(df_temp,word[0], label_col)])

            for label in df_temp[label_col].value_counts().index.tolist():
                pmi_label_dict[label].append([word[0]]+[self._pmiIndivCal(df_temp,word[0],label,label_col)])

        pmidf = pd.DataFrame(pmilist)
        pmidf.columns = ['word','pmi']

        for label in df_temp[label_col].value_counts().index.tolist():
            pmi_label_dict[label] = pd.DataFrame(pmi_label_dict[label])
            pmi_label_dict[label].columns = ['word','pmi']

        return pmi_label_dict, pmidf
    
    def get_pmi_dict(self, df, topk, label_col):
        pmi_dict, pmidf = self._pmiForAllCal(df, topk, label_col)
        return pmi_dict
        
    def generate_wordcloud(self, job_title_encode, topk_temp_both):

        for list_tuples in topk_temp_both:
            test_list_tuples_dict = dict(list_tuples)
            wordcloud = WordCloud(width=900, height=900, background_color ='white')
            wordcloud.generate_from_frequencies(test_list_tuples_dict)
            plt.figure(figsize = (8, 8), facecolor = None) 
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad=2)
            save_fig_name = "wordcloud_" + str(job_title_encode) + ".png"
            plt.savefig(save_fig_name)
            plt.show()
    
    def _add_cluster_label_each_title(self, df_tools, job_title, job_group_col, number_of_clusters, job_description_col, url_col):
        '''Adds cluster labels to df'''
        
        pd.options.mode.chained_assignment = None

        #take filtered df by title
        df_ds = df_tools[df_tools[job_group_col] == job_title]
        df_ds.reset_index(drop = True, inplace = True)
        job_description_values = df_ds[job_description_col].values
        
        #initialize topk of only tool_features
        tool_features_list = df_tools.columns.tolist()
        topk_tf_idf_single, topk_tf_idf_phrase =         self.fe.get_subset_counter_list(job_description_values, tool_features_list, tool_features_list)  
        topk_full_idf = topk_tf_idf_single + topk_tf_idf_phrase

        #only model using tool features
        feature_list = [a_tuple[0] for a_tuple in topk_full_idf]
        df_ds_subset = df_ds[feature_list]
        
        #initialize clustering model
        model = SpectralClustering(n_clusters = number_of_clusters, random_state=23, n_init = 100, affinity='rbf')
        model.fit(df_ds_subset)
        
        #put cluster label back in df
        df_label = pd.DataFrame(model.labels_)
        df_label.reset_index(drop = True, inplace = True)
        df_label.columns = ["label"]
        df_ds_subset["cluster_label"] = df_label["label"].copy()
        df_ds_subset[job_description_col] = df_ds[job_description_col].copy()
        df_ds_subset[url_col] = df_ds[url_col].copy()

        #build a temporary dictionary of cluster_number and cluster_name
        df_temp_only_labels = pd.DataFrame(df_ds_subset["cluster_label"].value_counts()).reset_index()
        df_temp_only_labels.columns = ["label", "number_of_entries"]

        df_label_dict = {}
        n_clusters = df_temp_only_labels.shape[0]

        for i in range(0,n_clusters):
            df_label_dict[i] = df_temp_only_labels.loc[i, "label"]

        return df_ds_subset, df_label_dict, topk_tf_idf_single, topk_tf_idf_phrase
          
    
    def _get_high_pmi_words_each_cluster(self, df_label_dict, topk_tf_idf_single, topk_tf_idf_phrase,                           df_ds_subset, job_description_col):
        
        '''Returns dictionary of top pmi words for each cluster_label '''
        
        #general features
        general_features = ["cluster_label", job_description_col]

        #only model using tool features
        feature_list_single = [a_tuple[0] for a_tuple in topk_tf_idf_single]
        feature_list_phrase = [a_tuple[0] for a_tuple in topk_tf_idf_phrase]

        #make single and phrase dfs
        df_ds_subset_single = df_ds_subset[general_features + feature_list_single]
        df_ds_subset_phrase = df_ds_subset[general_features + feature_list_phrase]

        #get pmi for each cluster number single
        pmi_dict_single = {}
        pmi_dict_phrase = {}
        pmi_dict_single, pmidf_single = self._pmiForAllCal(df_ds_subset_single, topk_tf_idf_single, "cluster_label")
        pmi_dict_phrase, pmidf_phrase = self._pmiForAllCal(df_ds_subset_phrase, topk_tf_idf_phrase, "cluster_label")

        #get top pmi words each for single and phrase
        topk_tf_idf_all = topk_tf_idf_single + topk_tf_idf_phrase
        high_mi_scores_each_cluster_dict = {}
        topk_all_dfs = []

        for cluster_number, cluster_name in df_label_dict.items():
            #get topwords for each cluster
            high_mi_words_list_single = pmi_dict_single[cluster_number].sort_values('pmi',ascending=0)["word"][0:10].tolist()
            high_mi_words_list_phrase = pmi_dict_phrase[cluster_number].sort_values('pmi',ascending=0)["word"][0:10].tolist()
            high_mi_scores_each_cluster_dict[cluster_number] = high_mi_words_list_single + high_mi_words_list_phrase
        
        #returned dictionary is of structure {0: ["a","b"], 1:["c","d"]...}
        return high_mi_scores_each_cluster_dict
    
    def _get_distinct_terms_each_title(self):
        '''create a flat dictionary of all terms for each title''' 
        for max_clusters, dict_items in self.top_words_by_cluster_dict.items():       
            self.topwords_by_title_dict[max_clusters] = {}
            for job_title_key, top_words_dict in dict_items.items():
              #in this level, you see all clusters within job_title
                temp_list = []               
                for cluster_number, top_pmi_words in top_words_dict.items():      
                    if cluster_number != max_clusters:
                          temp_list.extend(top_pmi_words[0:self.number_words_each_cluster])
                    #this is used in nodes_list. Plot all nodes if it represents top words
                    elif cluster_number == max_clusters:
                          temp_list.extend(top_pmi_words[0:])
                self.topwords_by_title_dict[max_clusters][job_title_key] = temp_list

    
    def get_distinct_terms_each_cluster(self, df_tools, job_titles_list, job_group,                                         number_of_clusters_upto, last_dict_element,                                         job_description_col, url_col, top_tools_dict):  
        
        '''get distinct terms and store in a dictionary'''
        for number_of_clusters in range(2, number_of_clusters_upto+1):
            
            self.top_words_by_cluster_dict[number_of_clusters] = {}
            
            for job_title in job_titles_list:
                df_ds_subset, df_label_dict, self.topk_single, self.topk_phrase =                 self._add_cluster_label_each_title(df_tools, job_title, job_group,  number_of_clusters, job_description_col, url_col)
                
                self.top_words_by_cluster_dict[number_of_clusters][job_title] =                 self._get_high_pmi_words_each_cluster(df_label_dict, self.topk_single,                                                       self.topk_phrase, df_ds_subset, job_description_col)

                #last element of this dictionary represents the topwords
                self.top_words_by_cluster_dict[number_of_clusters][job_title][number_of_clusters] = top_tools_dict[job_title].copy()
         
                #write cluster labels in df for the app
                df_cluster = df_ds_subset[["cluster_label", url_col]].copy()
                parsed_job_title = job_title.lower().replace(" ", "_")
                col_name = "cluster_label_" + str(parsed_job_title) + "_" + str(number_of_clusters)
                df_cluster.columns = [col_name, url_col]
                self.df_tools_with_clusters = self.df_tools_with_clusters.merge(df_cluster, on = url_col, how = "left")
                
        self._get_distinct_terms_each_title()
          
        
    def _get_edges_df(self, df_tools_temp, job_title, clean_job_title_col, url_col, job_title_col,                                  cluster_label_col, job_description_col,                                  top_words_by_cluster_dict, last_dict_element):

        '''Makes a word co-occurence dataframe from job descriptions'''
        
        df_tools_subset = df_tools_temp[df_tools_temp[clean_job_title_col] == job_title]
        number_of_records = df_tools_subset.shape[0]
        df_tools_subset.drop([url_col, job_title_col, clean_job_title_col, cluster_label_col], inplace = True, axis = 1)
        df_temp = df_tools_subset.copy()
        df_temp = df_temp.set_index(job_description_col)
        df_asint = df_temp.astype(int)
        coocc = df_asint.T.dot(df_asint)
        coocc.values[np.tril(np.ones(coocc.shape)).astype(np.bool)] = 0
        a = coocc.stack()
        a = a[a >= 1].rename_axis(('word_1', 'word_2')).reset_index(name='cooccur_count')
        a.sort_values(by = "cooccur_count", ascending = False, inplace=True)
        a.reset_index(inplace = True, drop = True)

        return a   
    
    
    def _calculate_chi2(self, word_1_count,word_2_count,cocc_count, number_of_records):
        
        #table_i_j where i represents row, j represents column  
        table_1_1 = cocc_count
        table_2_1 = word_1_count - cocc_count
        table_1_2 = word_2_count - cocc_count
        table_2_2 = number_of_records - (table_2_1 + table_1_2 + cocc_count)
        # contingency table       
        table = [[table_1_1, table_2_1], [table_1_2, table_2_2]]
        
        #calculate chi2 with yates correction
        c, p, dof, expected = chi2_contingency(table)    
        return np.round(p,3)
    
    def _add_edges(self, df_tools, clean_job_title_col, url_col, job_title_col,                      cluster_label_col, job_description_col):    
        '''adds edges to the network graph'''
        
        for max_clusters, dict_items in self.topwords_by_title_dict.items():
            self.df_edge_dict[max_clusters] = {}
            
            for job_title, nodes_list in dict_items.items():
                
                #calculate laplace smoothing
                df_title = df_tools[df_tools[clean_job_title_col] == job_title]
                number_of_records = df_title.shape[0]
                job_description_values = df_title[job_description_col].values
                counter_single = Counter()
                description = df_tools["job_description"].values
                
                #make counter for all single words
                for review in job_description_values:
                    counter_single.update([word.lower() 
                                  for word 
                                  in re.findall(r'\w+', review) 
                                  if word.lower() not in self.fe.stop and len(word) > 3])
                    
                #apply laplace smoothing
                word_count = 0
                for i,value in dict(counter_single).items():
                    word_count = word_count + value
                total_word_count_laplace_smoothing = word_count + len(counter_single)
                
                #get word count only for words/phrases in nodes_list                     
                topk_single, topk_phrase =                 self.fe.get_subset_counter_list(job_description_values, nodes_list, nodes_list)

                #Find out if a particular review has the word from topk list
                freqReview = []
                topk = topk_single + topk_phrase
                for i in range(len(job_description_values)):
                  # you feed a list of elements in counter ["a", "a", "c", "r"] then counter returns counter object {"a":2, "c":1, "r":1}
                    tempCounter = Counter([word.lower() for word in re.findall(r'\w+',job_description_values[i])])
                    topkinReview = [1 if tempCounter[word] > 0 else 0 for (word,wordCount) in topk]
                    freqReview.append(topkinReview)

                #Prepare freqReviewDf
                freqReviewDf = pd.DataFrame(freqReview)
                dfName = []
                for c in topk:
                    dfName.append(c[0])
                freqReviewDf.columns = dfName
                self.freqReviewDf = freqReviewDf
                
                #additional may have to delete
                self.temp_dict = dict(topk_single + topk_phrase)
                
                
                #get coocurence matrix in the form of an edge dataframe for the network graph
                df_edge = self._get_edges_df(df_tools, job_title, clean_job_title_col, url_col, job_title_col,                                  cluster_label_col, job_description_col,                                  self.top_words_by_cluster_dict, max_clusters)
                
                df_edge = df_edge[(df_edge['word_1'].isin(nodes_list)) & (df_edge['word_2'].isin(nodes_list))]

                #apply word counts with laplace smoothing to edge df
                df_edge["word_1_count"] = df_edge["word_1"].apply(lambda x: self.freqReviewDf[str(x)].sum() + 1)
                df_edge["word_2_count"] = df_edge["word_2"].apply(lambda x: self.freqReviewDf[str(x)].sum() + 1)
                df_edge["cooccur_count"] = df_edge["cooccur_count"].apply(lambda x: x + 1)                
                df_edge["pmi_temp"] = total_word_count_laplace_smoothing *                                         (df_edge["cooccur_count"]) / (df_edge["word_1_count"] * df_edge["word_2_count"])
                df_edge["pmi"] = df_edge["pmi_temp"].apply(lambda x: np.round(math.log(x,2),0))
                
                df_edge['p_value_chi2'] = df_edge.apply(lambda row : self._calculate_chi2(row['word_1_count'], 
                     row['word_2_count'], row['cooccur_count'], number_of_records), axis = 1)
                
                df_edge["pmi"] = df_edge["pmi"].apply(lambda x: 0 if x < 0 else x)
                
                
                df_edge.loc[df_edge["cooccur_count"] < 5, 'pmi'] = 0
                df_edge.drop("pmi_temp", inplace = True, axis = 1)
                
                self.df_edge_dict[max_clusters][job_title] = df_edge.copy()
                
                tuple_list = list(zip(df_edge["word_1"], df_edge["word_2"], df_edge["pmi"], df_edge["p_value_chi2"]))
                
                significance_value = 0.05
                
                for tuple_item in tuple_list:
                    
                    named_title = "pmi:" + str(tuple_item[2]) + ", p-value:" + str(tuple_item[3])
                    
                    if (tuple_item[3] < significance_value) & (tuple_item[2] != 0) :
                        self.net_dict[max_clusters][job_title].add_edge(tuple_item[0],tuple_item[1],                                                                         title = named_title, physics = False,                                                                       width = 0.5)
                    
                    else:
                        self.net_dict[max_clusters][job_title].add_edge(tuple_item[0],tuple_item[1],                                                                         title = named_title, physics = False,                                                                        width = 0.0005)
                        

    def _add_nodes(self):
        '''add nodes to the network graph'''
        
        #number of words per cluster
        top_words_color = "#ea96a3"
        color_dict = {0:"#d7944e", 1:"#4eabb7", 2:"#49ae83", 3: "#ab9e47", 4:"#bea4ea"} 

        #check out color schemes!: https://www.w3schools.com/colors/colors_analogous.asp
        
        for max_clusters, dict_items in self.top_words_by_cluster_dict.items():
            self.net_dict[max_clusters] = {}
            

            #initialize nodes in graph
            for job_title, cluster_contents_dict in dict_items.items():
                self.net_dict[max_clusters][job_title] = Network(height="500px", width="100%", font_color="black",heading='')
                self.net_dict[max_clusters][job_title].force_atlas_2based()

                for cluster_label, cluster_top_words_list in cluster_contents_dict.items():

                  #add nodes. if it is last_element, get all words, otherwise, get specified number only
                    if cluster_label == max_clusters:
                        nodes_list = cluster_top_words_list[0:]
                        nodes_length = len(nodes_list)
                        color = top_words_color
                        cluster_title = "Top Hard Skill"

                    else: 
                        nodes_list = cluster_top_words_list[0:self.number_words_each_cluster]
                        nodes_length = len(nodes_list)
                        color = color_dict[cluster_label]
                        cluster_number = cluster_label + 1
                        cluster_title = "Cluster " + str(cluster_number)

                    #title_list appears on hover
                    title_list = []
                    color_list = []

                    #just makes a list of repeated color and cluster_title names for pyvis
                    for i in range(0, nodes_length):
                        title_list.append(cluster_title)
                        color_list.append(color)

                    self.net_dict[max_clusters][job_title].add_nodes(nodes_list, title=title_list, 
                              color=color_list)
        
    def generate_network_graph(self, df_tools, label_col, url_col, job_title_col, encoded_job_title_col, description_col):
        self._add_nodes()
        self._add_edges(df_tools, label_col, url_col, job_title_col, encoded_job_title_col, description_col)
        


# In[ ]:




