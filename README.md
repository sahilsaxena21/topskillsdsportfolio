# Characterizing Industry Needs in Data Science

## Executive Summary

Data Science is a broad discipline encompassing a wide range of tools, languages , technologies and frameworks. Aspiring data scientists in Canada looking to upskill may want to ensure that the skills they are practicing actually align with current industry needs. If this process is not programmatic , the user would have to read through the job descriptions manually, while keeping a record of the skills enlisted in each one, which takes time and effort. Moreover, data science is an evolving field, and hence this process would need to be repeated every so often to stay current.

The app attempts to allivieate this problem by taking an **programamtic** approach. Hundreds of job postings are scraped from Indeed.ca. The in-demand hard-skills are also scraped from well-maintained webpages such as O'Reilly's Data Science Glossary, Google's Machine Learning Glossary and more. Both datasets are processed and cross-referenced for insights, thereby mitigating need for manual reading of job postings.

Refer to the Streamlit app here: https://share.streamlit.io/sahilsaxena21/topskillsdsportfolio/main/app.py

## Methodology

The following analytical approach was used to extracts insights visualized in the Streamlit app.

![Methodology](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/image_files/methodology.JPG)


## Key Insights

![Hard Skills Dependency Graph of Records with Job Title of ‘Data Scientist’](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/image_files/ds_insights_graph.png)
<sub>this is a test</sub>

<sub>1 based on a sample of 163 job postings scraped from Indeed.ca between Nov. 2020 – Feb. 2021.</sub>
<sub>2 nodes in pink illustrate the 10 most commonly occurring hard skills. Other node colors represents the clusters or groupings of job postings from their job descriptions, as segregated by the spectral clustering algorithm. Refer to code for details on hyperparameters and other algorithm implementation details.</sub>
<sub>3 Cluster node labels are the top 5 terms that best characterize a cluster, ranked based on Laplace Smoothed Positive Pointwise Mutual Information (PPMI)</sub>

As per the above figure, current industry needs can largely be classified into 3 types of data scientists. These are inferred from the figure as below:

1.	The “full-stack” data scientist roles with wide-ranging skills involving data **cleaning** (using **pandas**), creating **pipelines** and **interpreting** findings using visualization tools such as **tableau**
2.	Roles that recognize use of **Kotlin**. Kotlin is a Java based programming language, growing rapidly to improve upon the current deficiencies of the programming language Scala.
3.	Two more **research-based** roles found to be distinctly separated into two categories:
    1. **Graph** and **Reinforcement** learning focussed roles with ability to develop algorithms from scratch
    2. b.	**Deep Learning** focussed roles with skills in developing deep neural networks (with LSTM found to be most sought after) along with other machine learning related know-how
