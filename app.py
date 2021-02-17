#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

import numpy as np
import seaborn as sns

import streamlit as st
import streamlit.components.v1 as components  

# Data Viz Pkgs
import matplotlib.pyplot as plt
# %matplotlib inline
# import matplotlib
# matplotlib.use('Agg')
import os

from PIL import Image
from IPython.core.display import display, HTML

html_temp = """
            <div style="background-color:{};padding:10px;border-radius:10px">
            <h1 style="color:{};text-align:center;">Exploring Industry Needs in Data Science App </h1>
            </div>
            """


# In[3]:


st.cache(persist=True)
def load_data():
    df_tools = pd.read_parquet("data/df_tools_with_clusters.parquet")
    df_edge_dict = np.load('data/df_edge_dict.npy',allow_pickle='TRUE').item()
    return df_tools,df_edge_dict
df_tools,df_edge_dict = load_data()


# In[4]:


def plot_countplot(title, clusters):

    parsed_job_title = title.lower().replace(" ", "_")
    col_name = "cluster_label_" + str(parsed_job_title) + "_" + str(number_of_clusters)
    
    x = df_tools[col_name].value_counts()
    
    # Bar chart
    labels = ["Cluster " + str(i) for i in range(1,x.shape[0]+1)]
    y_pos = np.arange(len(labels))
    sizes = x.values.tolist()
    #colors
    all_colors = ["#d7944e","#4eabb7","#49ae83", "#ab9e47","#bea4ea"]
    colors = [all_colors[i] for i in range(0,x.shape[0])]
    
    #make temp dataframe
    df_temp = pd.DataFrame({'cluster_number': labels, 'number_of_records': sizes})

    fig,ax1 = plt.subplots(1, figsize=(10,5))
    plt.title("Number of Records By Cluster")
    splot = sns.barplot(x = "cluster_number", y = "number_of_records", data = df_temp, ax = ax1, palette = colors)
    
    plt.xlabel("Cluster")
    plt.ylabel("Number of Records")
    plt.show()
    st.pyplot(fig)
    


# In[5]:


def plot_bar_chart(df_tools, title):
    top_tools_dict = {}
    df_tools_ds = df_tools[df_tools["job_group"] == title]

    cols_to_drop = []
    for column_name in df_tools_ds.columns.tolist():
        if "cluster_label" in column_name:
            cols_to_drop.append(column_name)
    general_cols = ["job_title", "url", "job_description", "job_group", "encoded_job_title"]
    cols_to_drop = cols_to_drop + general_cols
    df_tools_ds = df_tools_ds.drop(cols_to_drop, axis = 1)
    term_frequency_series = df_tools_ds.sum(axis = 0).sort_values(ascending = False)
    df_term_frequency = pd.DataFrame(term_frequency_series)
    df_term_frequency.reset_index(inplace = True, drop = False)
    df_term_frequency.columns = ["tool", "count"]
    top_tools_dict[title] = df_term_frequency["tool"][0:11].tolist()
    df_term_frequency["ratio"] = df_term_frequency["count"] / df_term_frequency.shape[0] * 100

    #plotting
    fig,ax1 = plt.subplots(1, figsize=(10,8))
    plt.title(title)
    sns.barplot(x = "ratio", y = "tool", data = df_term_frequency.iloc[0:11, :] , orient = 'h', ax = ax1)
    plt.xlabel("Term Occurence Frequency (Percent)")
    plt.ylabel("Hard Skills")
    st.pyplot(fig)


# In[6]:


menu = ["Home", "About"]
choice = st.sidebar.selectbox("Menu",menu)
st.markdown(html_temp.format('royalblue','white'),unsafe_allow_html=True)


# In[7]:


if choice == "About":
    st.markdown("")
    st.markdown("This app attempts to shed light on the current industry needs in Canada in the broad discipline of Data Science.")
    st.markdown("Scraped data Sources")
    st.markdown("1) Indeed.ca job postings (between Nov 2020 - Jan, 2021) with titles 'data scientist', 'machine learning engineer', 'data engineer' or 'data analyst'")
    st.markdown("2) Hard skills from Wikipedia's Glossary of ML, Google's Machine Learning Glossary, datascienceglossary.org and more.")
    st.markdown("Hope this helps you understand industry needs better!")
    st.markdown("")
    st.markdown('''Designed by: **Sahil Saxena**''')
    st.markdown("This project is licensed under the terms of the MIT license.")

    
elif choice == "Home":
    
    #header 1
    st.header('What are the key words associated with each job title?')
    
    with st.beta_expander('How is this constructed?'):
        st.write("Word clouds are populated programatically using Pointwise Mutual Information as a measure of the specific association of words in job descriptions with each job title. Text size represents relative occurence of the term or phrase")

        
    col1, col2 = st.beta_columns(2)
    with col1:
        st.header("Data Scientist")
        st.image("image_files/wordcloud_0.png", use_column_width=True)

    with col2:
        st.header("Machine Learning Engineer")
        st.image("image_files/wordcloud_2.png", use_column_width=True)

    col1, col2 = st.beta_columns(2)
    with col1:
        st.header("Data Analyst")
        st.image("image_files/wordcloud_1.png", use_column_width=True)

    with col2:
        st.header("Data Engineer")
        st.image("image_files/wordcloud_3.png", use_column_width=True)        
        

        
    #header 2
    st.markdown('Choose a Job Title Below to Explore its Industry Needs')    
    title=st.selectbox('Select Job Title',('Data Scientist','Data Engineer', 'Machine Learning Engineer', 'Data Analyst'))
    
    number_of_records = df_tools[df_tools["job_group"] == title].shape[0]
    st.markdown("")
    st.markdown(f"Number of Records in Database: {number_of_records}")    
    st.header('What are the top hard skills?')
    with st.beta_expander('How is this constructed?'):
        st.write("A list of terms used in data science were scraped from various websites including Google's Machine Learning Glossary, Wikipedia and more. These are represented as 'hard skills' in this exploratory tool")
    
    plot_bar_chart(df_tools, title)

    #header 3
    st.header('How can we classify postings within the job title?')
    number_of_clusters = st.selectbox('Choose Number of Clusters',(2,3,4,5))
    
    #header 4
    st.header(f"{title} Dependency Graph")
    
    with st.beta_expander('How is this constructed?'):
        st.write("We first apply a clustering algorithm to see if job postings can be grouped in a certain way based on the hard skills in the job descriptions. These are the color coded cluster nodes in the dependency graph. To help with understanding cluster nodes in relation to top hard skills, the graph also shows the top hard skills in pink.")
        st.write("Cluster nodes displayed on the graph are selected with some care. Each cluster node characterizes the trait attributes of each cluster the best. Hence, when taken collectively, the nodes for each cluster actually depicts what the cluster represents. This is also done programitically through the use of PMI")
        st.write("We then construct a dependency graph. A dependency graph shows the co-occurence of hard skills in a single job posting. Hence, hard skills that co-occur many times can be thought of as complementary and/or substitutive skills, depending on two skills being compared.")
        st.write("The extent of co-occurence is measured using two metrics. Firsly, we calculate laplace smoothed positive pointwise mutual information (PPMI). This provides a measure of association of two terms after normalizing it with the frequency of each term. Secondly, we observe the statistical significance of independence of the given two terms using Chi2 test with Yates correction for further stringency. All terms that have co-occured atleast 5 times, with a default significance level of 0.05 are illustrated with bolded edges.")    
        st.write("If you like, you can play around with other significance levels, and display the resultant co-occurence dataframe below")    
    
    parsed_string = title.lower().replace(" ", "_")
    plot_name = str(number_of_clusters) + "_" + str(parsed_string) + ".html"
    file_path = "html_files/" + plot_name
    HtmlFile = open(file_path, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 1000,width=800)
    
    #countplot
    st.header("Cluster Info")
    
    col1, col2 = st.beta_columns(2)
    with col1:
        st.header("Cluster Composition")
        plot_countplot(title, number_of_clusters)

    with col2:
        st.header("Elbow Curve")
        image_file_path = "image_files/" + title + ".png"
        st.image(image_file_path, use_column_width=True)
        

    if st.checkbox("Click to View the Co-occurence Dataset",False):
            chosen = st.radio("Dataset Filtered by Chi2 Significance",("0.05","0.01","0.001"))
            significance = float(chosen)
            df_edge_subset = df_edge_dict[number_of_clusters][title]
            df_edge = df_edge_subset[df_edge_subset["p_value_chi2"] <= significance]
            "dataset", df_edge
  
    


# In[ ]:





# In[ ]:




