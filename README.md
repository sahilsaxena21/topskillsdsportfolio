# Programmatic Characterization of Skill Needs in Canada's Data Science Industry


## Scripts
The analysis is taken up by 4 python scripts and 4 python classes as outlined below:

1. [01_Scraper.py](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/_01_Scraper.py) contains two classes, the JobPostingsCollector and HardSkillsCollector to create the datasets for this analysis. Both the scrapers are written in Python using the web interaction library Beautiful Soup.
2. [02_FeatureExtractor.py](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/_02_FeatureExtractor.py) contains Feature_Extractor_and_Processor, a helper class for the ModelContainer class to process the job descriptions, conduct feature extraction and selection.
3. [03_ModelContainer.py](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/_03_ModelContainer.py) contains the Modeling_and_Visualization class which contains code for various statistical and machine learning models used in the analysis including Pointwise Mutual Information (PMI), spectral clustering, and the network graphs.
4. [04_SaveFinalFiles.py](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/_04_SaveFinalFiles.py) calls on the Modeling_and_Visualization class to apply the models, and then save the results locally. 


## Executive Summary

Refer to the Streamlit app here: https://share.streamlit.io/sahilsaxena21/topskillsdsportfolio/main/app.py

Data Science is a rapidly booming industry, attracting talent from a variety of educational backgrounds. It is also an inter-disciplinary one, because in practice, it utilizes techniques and theories drawn from many fields within the context of mathematics, statistics, computer science, domain knowledge and information science. Some of the more common job titles within this industry includes data scientist, machine learning engineer, data analyst and data engineer. However, due to a lack of standardization of these job roles, the set of skills that distinguish one job title from another is not often clear. Moreover, given the broad set of tools, technologies and frameworks in this industry, it is not uncommon for two jobs to require a different set of skills, even if the job titles are exactly the same. Hence, aspiring individuals who are new to data science, but are looking to enter into this industry may find it confusing to do so. These individuals want to identify what skills are in-demand, so they can take appropriate steps to up-skill to improve their candidacy and break into this industry in the Canadian context.

This exploratory study aims to provide insights to such users at the fundamental level. We use job descriptions in online job postings as an indicator of the baseline set of skill requirements sought after by employers. An ML pipeline is developed that scrapes job postings from Indeed.ca (N=454), processes these descriptions, and delivers insights to users programmatically. Although aspiring individuals looking to enter this industry typically already have access to such postings and can take this process up by themselves, if this process is not programmatic, the user would have to read through the job descriptions manually, while keeping a record of the skills enlisted in each one, which takes time and effort. Moreover, data science is an evolving field, and hence this process would need to be repeated every so often to keep the results current.


## Analytics Problem Framing
We break down the problem into Analytical Questions (AQs) to be informed by our app. These are as follows: 
* AQ 1: Amongst the several job titles in the data science industry (i.e. 'data scientist', 'data analyst', 'machine learning engineer' and 'data engineer'), which job title is most relevant to me? In other words, how does one job title distinguish from the other?
* AQ 2: What hard skills are the most sought after? In other words, which languages, tools, technologies and frameworks occur most frequently in job postings for each job title?
* AQ 3: Given the variance in roles within each job title, what are the common types of roles within each job title so I can hone down and up-skill for that particular role? Which type is the most in-demand?
* AQ 4: How can I leverage my current skillset, so I up-skill in the most efficient way to break into this field?


## Methodology

The following analytical approach was used to extracts insights visualized in the Streamlit app.

![Methodology](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/image_files/methodology.png)


### 1. Data Collection

Two datasets are scraped for this analysis.
The first is a sample of N=454 job postings scraped from Indeed.ca from November 18, 2020 to February 20, 2021. The dataset contains four columns, the url of the job posting, the job title of the posting, the job description of the posting (i.e. all content within the job posting) and another job title column which buckets the job posting into one of the four categories (i.e. 'data scientist', 'data analyst', 'machine learning engineer' or 'data engineer'). All postings that did not contain any of the above terms in their job title were not included in the sample.
Secondly, the dataset of hard skills (i.e. comprised of languages, frameworks, tools and technologies) used in data science is scraped from [AnalyticsVidhya] (https://www.analyticsvidhya.com/glossary-of-common-statistics-and-machine-learning-terms/), [DataScienceGlossary](http://www.datascienceglossary.org/), [Google Developer's Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/), [Wikipedia's Glossary of Artificial Intelligence](https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence%22) and [Wikipedia's list of programming languages](https://en.m.wikipedia.org/wiki/List_of_programming_languages). A total of _**N=1,770**_ terms were scraped from the above web pages including terms such as **reinforcement learning, hive, flume, pandas and python**.


### 2. Feature Engineering
TF-IDF term weighting is used to extract the meaningful tokens (i.e. monograms and bigrams) from job descriptions as features. Then, Pointwise Mutual Information (PMI) is used as a measure of association between each token to the job title. A token that has a higher PMI value for a given job title indicates a higher probability of occurrence of the token in the job description for a given job title relative to all other job titles. Hence, this methodology is used to inform **AQ 1** i.e. to identify the terms that distinguish one job title from another. This is then presented as word clouds. Word clouds includes the top 30 tokens with the highest PMI associated with ach job title as illustrated below.


![Top Terms](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/image_files/wordcloud_all.png)


A second feature selection step is performed to shortlist the features to the hard skills using the hard skills data base. These features are boolean encoded. Then a simple categorical bar plot is made plotting the hard skill to its occurrence frequency (in percent) in job postings within each title. This plot informs **AQ 2** i.e. to identify the top hard skills for each job title. An illustration for job title ‘data scientist’ is provided below.

![Top Skills](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/image_files/hardskills.JPG)


### 3. Clustering and Cluster Interpretation
A graph based method (i.e. spectral clustering) is used to group job postings based on the hard skills mentioned in the job descriptions. A graph based method is used because of its robustness in high dimensionality compared to Eucleadian methods (e.g. k-means) because it uses **distance on the graph** (e.g. the number of shared neighbours) which is more meaningful in high-dimensions. Hence, if two job postings belong to the same cluster, it signifies that there is a **considerable overlap** in the set of skills mentioned in their job description.

We then identify the skills that **distinguish** one cluster from another by **identifying the trait characteristics of each cluster**. A similar approach is taken as in Step 2 i.e. using PMI, but using a modified version of this called laplace-smoothed positive pointwise mutual information (PPMI) to make the measure more robust to infrequent events. Again, the skill that has a higher PMI value for a given cluster indicates a higher probability of occurrence of the skill in the given cluster relative to all other clusters. The top 5 skills with the highest PPMI values are selected to represent each cluster. In this way, the approach informs **AQ 3** i.e. to **characterize** the different types of roles within each job title. This is then visualized as nodes on a skills dependency network graph as described further in the Step below. 



### 4. Co-occurence Significance Test

The non-parametric chi-squared test is used to identify the skills that significantly co-occur in a job posting using the occurrence frequency contingency table. If a given pair of skills is found to be statistically significant under this test, it signifies that the pair of skills is unlikely to be independent of each other. In other words, their co-occurence in a job posting is unlikely to be due to random chance. It is to be noted that a pair of skills found to be statistically significant could be interpreted by the user as either being **substitutive** (e.g. python and r) or **complementary** (e.g. python and sql) depending on the skills being compared. By identifying the skills that significantly co-occur together, **users can understand the complementary skills**, and consequently, make data-informed decisions on their upskill directions, in a way that leverages their existing skillset to improve their candidacy in this field (**AQ 5**).


## Key Insights
<sub>for better image clarity, refer to the pdf here https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/Insights%20Interpretation%20Summary.pdf</sub>

![Hard Skills Dependency Graph](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/image_files/ds_insights_graph.png)

![Interpretation](https://github.com/sahilsaxena21/topskillsdsportfolio/blob/main/image_files/dstypes.JPG)

**Hard Skills Dependency Graph of Records with Job Title of ‘Data Scientist’ [1, 2, 3]**

* <sub>[1] based on a sample of _N=163_ job postings scraped from Indeed.ca between Nov. 2020 – Feb. 2021.</sub>
* <sub>[2] nodes in pink illustrate the 10 most commonly occurring hard skills. Other node colors represents the clusters or groupings of job postings from their job descriptions, as segregated by the spectral clustering algorithm. Refer to code for details on hyperparameters and other algorithm implementation details.</sub>
* <sub>[3] cluster node labels are the top 5 terms that best characterize a cluster, ranked based on Laplace Smoothed Positive Pointwise Mutual Information (PPMI)</sub>

As per the above network graph, current industry needs can largely be seperated into 3 types of data scientists. These are inferred from the figure as below:

1.	The “full-stack” data scientist roles with wide-ranging skills involving data **cleaning** (using **pandas**), creating **pipelines** and **interpreting** findings using visualization tools such as **tableau**
2.	Roles that recognize use of **Kotlin**. Kotlin is a Java based programming language, growing rapidly to improve upon the current deficiencies of the programming language Scala.
3.	Two more **research-based** roles found to be distinctly separated into two categories:
    1. **Graph** and **Reinforcement** learning focussed roles with ability to develop **algorithms** from **scratch**
    2. **Deep Learning** focussed roles with skills in developing deep neural networks (with **LSTM** found to be most sought after) along with **other** machine learning related know-how
