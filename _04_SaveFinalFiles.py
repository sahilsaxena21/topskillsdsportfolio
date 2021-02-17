#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import modelcontainer to initalize models. modelcontainer internally calls the featureextractor class
from _03_ModelContainer import Modeling_and_Visualization
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt


# In[2]:


# initialize modelcontainer object
job_title_col = "job_title"
url_col = "url"
job_description_col = "job_description"
label_col = "job_group"
word_col = "word"
encoded_job_title_col = "encoded_job_title"
indeed_file = "data/indeed.csv"
words_file = "data/words.csv"
number_words_each_cluster = 5

mc = Modeling_and_Visualization(job_title_col, url_col, job_description_col, label_col,                                      word_col, encoded_job_title_col, indeed_file, words_file, number_words_each_cluster)


# In[ ]:


#Save Wordclouds
number_of_pmi_words = 50
job_group = "job_group"
job_titles_list = ["Data Scientist", "Data Analyst", "Data Engineer", "Machine Learning Engineer"]
job_description_col = "job_description"
url_col = "url"
number_of_clusters_upto = 3
last_dict_element = number_of_clusters_upto
top_tools_dict = mc.fe.top_tools_dict

#plot and save word_cloud for each job_title
for job_title_encode in range(0,len(job_titles_list)):
    pmi_dict = mc.get_pmi_dict(mc.fe.df, mc.fe.topk_full, mc.fe.encoded_job_title_col)
    high_pmi_words = pmi_dict[job_title_encode].sort_values('pmi',ascending=0)["word"].tolist()[0:number_of_pmi_words]
    topk_temp_single, topk_temp_phrase = mc.fe.get_subset_counter_list(mc.fe.job_description, high_pmi_words,high_pmi_words)
    topk_temp_both = [topk_temp_single + topk_temp_phrase]
    mc.generate_wordcloud(job_title_encode, topk_temp_both)


# In[4]:


#Process for saving network graphs

#1. Perform clustering and get dict of top words for each cluster within each job_title
mc.get_distinct_terms_each_cluster(mc.fe.df_tools, job_titles_list,                                 job_group, number_of_clusters_upto, last_dict_element,                                 job_description_col, url_col, top_tools_dict)

#2. Build network graph of top words
mc.generate_network_graph(mc.fe.df_tools, mc.fe.label_col, mc.fe.url_col,                           mc.fe.job_title_col, mc.fe.encoded_job_title_col, mc.fe.description_col)

#3. Save graphs
for max_clusters, dict_items in mc.net_dict.items():
    for job_title, net in dict_items.items():
        parsed_string = job_title.lower().replace(" ", "_")
        plot_name = str(max_clusters) + "_" + str(parsed_string) + ".html"
        print(plot_name)
        net.show_buttons(filter_=["physics"])
        net.show(plot_name)


# In[5]:


#save elbow curve plots
for job_title in job_titles_list:
    
    #obtain subset df
    df_ds_subset, df_label_dict, topk_tf_idf_single, topk_tf_idf_phrase =    mc._add_cluster_label_each_title(mc.fe.df_tools, job_title, job_group, 3, job_description_col, url_col) 
    df_ds_subset_new = df_ds_subset.drop(["cluster_label", "job_description", "url"], axis = 1)
    
    sc = SpectralClustering(n_clusters = 5, random_state=23, n_init = 100, affinity='rbf')
    sc.fit(df_ds_subset_new)
    model = KElbowVisualizer(sc, timings = False)
    model.fit(df_ds_subset_new)
    filepath = "image_files/" + job_title + ".png"
    model.show(outpath=filepath)
    plt.close()


# In[6]:


#save dataframe and network graph dictionary locally for app
import numpy as np
mc.df_tools_with_clusters.to_parquet("data/df_tools_with_clusters.parquet")
np.save('data/df_edge_dict.npy', mc.df_edge_dict)

