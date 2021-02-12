#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


st.cache(persist=True)
def load_data():
    df_tools = pd.read_parquet("data/df_tools_with_clusters.parquet")
    df_edge_dict = np.load('data/df_edge_dict.npy',allow_pickle='TRUE').item()
    return df_tools,df_edge_dict
df_tools,df_edge_dict = load_data()


# In[8]:


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
    
#     for p in splot.patches:
#         splot.annotate(np.round(p.get_height(),0), 
#                        (p.get_x() + p.get_width() / 2., np.round(p.get_height(),0)), 
#                        ha = 'center', va = 'center', 
#                        xytext = (0, 9), 
#                        textcoords = 'offset points')
    
    plt.xlabel("Cluster")
    plt.ylabel("Number of Records")
    plt.show()
    st.pyplot(fig)
    


# In[9]:


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


# In[10]:


menu = ["Home", "About"]
choice = st.sidebar.selectbox("Menu",menu)
st.markdown(html_temp.format('royalblue','white'),unsafe_allow_html=True)


# In[11]:


# st.sidebar.markdown(' **Characterizing Industry Needs in Data Science** ')

if choice == "About":
    st.markdown("")
    st.markdown(''' 
    Data Science is a broad discipline spanning a broad range of tools, technologies and skillset. 

    This app attempts to help characterize industry needs in this discipline using job descriptions from online job postings.

    Job postings with titles "data scientist", "machine learning engineer", "data engineer" or "data analyst" were scraped on November 20, 2020 from Indeed.ca to create the dataset.

    Furthermore, a list of tools and technologies used in Data Science were scraped from a variety of sources including Wikipedia's Glossary of ML, Google's Machine Learning Glossary, datascienceglossary.org and more. These were cross-referenced back to the job descriptions to help with categorizing each job posting.                   

    Enjoy!

    Designed by: 
    **Sahil Saxena**  ''')

    
elif choice == "Home":
    
    #header 1
    st.header('What are the key words associated with each job title?')   
        
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
    plot_bar_chart(df_tools, title)

    #header 3
    st.header('How can we classify postings within the job title?')
    number_of_clusters = st.selectbox('Choose Number of Clusters',(2,3,4,5))
    
    #header 4
    st.header(f"{title} Dependency Graph")
    parsed_string = title.lower().replace(" ", "_")
    plot_name = str(number_of_clusters) + "_" + str(parsed_string) + ".html"
    file_path = "html_files/" + plot_name
    HtmlFile = open(file_path, 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height = 1000,width=800)

    if st.checkbox("Click to View the Co-occurence Dataset",False):
            chosen = st.radio("Dataset Filtered by Chi2 Significance",("0.05","0.01","0.001"))
            significance = float(chosen)
            df_edge_subset = df_edge_dict[number_of_clusters][title]
            df_edge = df_edge_subset[df_edge_subset["p_value_chi2"] <= significance]
            "dataset", df_edge
  
    
    #countplot
    st.header("Cluster Composition")
    plot_countplot(title, number_of_clusters)


# In[ ]:





# In[ ]:




