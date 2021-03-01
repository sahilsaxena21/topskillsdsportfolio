#!/usr/bin/env python
# coding: utf-8

# In[12]:


'''
This script contains two classes.
1: JobPostingsCollector: for crawling job postings
2: HardSkillsCollector: for scraping the hard skills
'''

__author__ = 'Sahil Saxena'
__email__ = 'sahil.saxena@outlook.com'

import pandas as pd
from urllib.request import Request, urlopen
import requests
import re
from bs4 import BeautifulSoup

class JobPostingsCollector:
    '''
    This class crawls and collects job descriptions from job postings. It then saves 
    the results locally in a csv file.
    '''

    def __init__(self, job_titles_list, job_title_col, url_col, job_description_col, label_col, num_pages = 10, existing_file = None):
        '''       
        Attributes
        ----------
        job_titles_list: list. list of job titles to include as search terms in Indeed.ca
        job_title_col: str. column name to be used for the job title as found in the job posting
        url_col: str. column name to be used for the url of the job posting
        job_description_col: str. column name to be used for the job description as found in the job posting
        label_col: str. column name to be used for the assigned job group. A job group is assigned from set 
                     {data scientist, machine learning engineer, data analyst, data engineer} depending 
                     on the title in the job posting
        num_pages: int. number of search pages to crawl in search results.
        existing_file: str. The location of the existing csv file, if one exists. If None, a new csv file will be created
        indeed_df: df. The final dataframe with the scraped job posting
        '''
        
        self.job_titles_list = job_titles_list
        self.job_title_col = job_title_col
        self.url_col = url_col
        self.job_description_col = job_description_col
        self.label_col = label_col
        self.num_pages = num_pages
        self.indeed_df = self._indeed_crawler(existing_file)

    def _load_data(self, filepath):
        '''Read a csv file from a provided directory. Returns a pandas dataframe'''
        return pd.read_csv(filepath, index_col=[0])
    
    def _write_file(self, df, filepath):
        '''Write a pandas dataframe to a csv file in the provided directory'''
        df.to_csv(filepath)
    
    def _add_job_group(self, x):
        '''
        Assigns a job group to a job posting based on the wording of the job title.
        
        Arguments
        ----------
        x: str. the job title string collected from the job posting
        
        Returns
        -------
        a job group in set {"Data Scientist", "Machine Learning Engineer", "Data Engineer","Data Analyst", "None"} 
        '''
        
        x = str(x).lower()
        if "data scien" in x:
            return "Data Scientist"
        elif "machine learning" in x:
            return "Machine Learning Engineer"
        elif "data engineer" in x:
            return "Data Engineer"
        elif "data analyst" in x:
            return "Data Analyst"
        else:
            return "None"
    
    def _clean_text(self, x):
        '''
        Search for all non-letters and replace with spaces
        
        Parameters
        ----------
        x: str. the text to be cleaned
        
        Returns
        -------
        letters_only: str. the processed text
        '''
        
        letters_only = re.sub("[^a-zA-Z]", " ",str(x))
        return letters_only
        
    def _indeed_crawler(self, existing_file):
        '''
        Crawls Indeed.ca job postings in a two-step process. 
        First, it gets the job posting URLs. 
        Then, it visits each URL to get the job descriptions.
        
        Parameters
        ----------
        existing_file: the location of the existing csv file, if one exists. If None, a new csv file will be created

        Returns
        -------
        a pandas dataframe with job_title, url, job_description and job_group columns
        '''
        
        #step 1: Collect urls and store results in dataframe
        df_urls = self._get_job_links_indeed()
        
        #step 2: Get job descriptions by visiting each url. Store results in a new dataframe
        df_new_scrape = self._get_job_descriptions_from_links(df_urls)
        
        #perform basic cleaning of job_description
        df_new_scrape[self.job_description_col] = df_new_scrape[self.job_description_col].apply(lambda x: self._clean_text(x))
                
        #if no existing dataset is provided, simply update the attribute with new scraped dataset
        if existing_file == None:
            file_path = "data/indeed.csv"
            self._write_file(df_new_scrape, file_path)
            return df_new_scrape

        #if this is an update to existing dataset, merge the newly scraped dataset with it
        elif existing_file != None:
            df_existing = self._load_data(existing_file)
            df_updated = df_existing.append(df_new_scrape, ignore_index=True)
            df_updated.drop_duplicates(inplace = True)
            df_updated.reset_index(drop = True, inplace = True)
            self._write_file(df_updated, existing_file)
            return df_updated     

    def _get_job_links_indeed(self):
        '''
        Crawls a given number of search result pages on Indeed.ca 
        Collects and stores the URLs of job postings.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        a pandas df with job title, job group and url columns
        
        '''        
        try:  
            dict_indeed = dict()
            
            for search_title in self.job_titles_list:
                for page_num in range(0, self.num_pages+1):    
                    search_title = str(search_title).replace(" ", "+").lower()
                    search_page_url = "https://ca.indeed.com/jobs?q=" + str(search_title) + "&l=Canada&jt=fulltime&start=" + str(page_num)
                    source = requests.get(search_page_url).text
                    soup = BeautifulSoup(source, "html.parser")

                    for match in soup.find_all("a", target= "_blank"):        
                        retrieved_title = (match.text).strip("\n").lower()
                        url = "https://ca.indeed.com" + str(match["href"])
                        dict_indeed[url] = retrieved_title

            df_urls = pd.DataFrame.from_dict(dict_indeed, orient="index", columns=[self.job_title_col])
            df_urls[self.url_col] = df_urls.index
            df_urls['index_col'] = range(1, len(df_urls) + 1)
            df_urls.index = df_urls["index_col"]
            df_urls.reset_index(drop = True, inplace = True)
            df_urls.drop("index_col", inplace = True, axis = 1)
            df_urls[self.label_col] = df_urls[self.job_title_col].apply(lambda x: self._add_job_group(x))
            df_urls = df_urls[df_urls[self.label_col] != "None"]          
            return df_urls
            
 
        except:
                print("Exceeded Daily Ping Limit. Please try again later")
    
    def _get_indeed_description(self, x):
        '''
        Parses the job descriptions by visting Indeed.ca webpage URLs.
        
        Parameters
        ----------
        x: str. the URL of the job posting
        
        Returns
        ----------
        description: str. the job description text within the job posting
        
        Raises
        ----------
        error if the ping limit has reached
        '''
        
        try:  
            source = requests.get(x).text 
            soup = BeautifulSoup(source, "html.parser")
            description = ""
            match = soup.find("div", class_= "jobsearch-jobDescriptionText")

            if match != None:
                description = match.text.strip("\n") 

            return description

        except:
            print("Exceeded Daily Ping Limit. Please try again later")
    
        
    def _get_job_descriptions_from_links(self, df_urls):
        '''
        Gets job descriptions from URLs.
        
        Parameters
        ----------
        df_urls: pandas df. the dataframe with the urls of the job postings but without the job descriptions
        
        Returns
        ----------
        df_updated: pandas df. the dataframe with the added job_description_col column that contains the 
                    job description of each url
        '''
        df_updated = df_urls.copy()
        df_updated[self.job_description_col] = df_updated[self.url_col].apply(lambda x: self._get_indeed_description(str(x)))
        return df_updated

if __name__ == '__main__':

    #define input columns
    job_title_col = "job_title"
    url_col = "url"
    job_description_col = "job_description"
    label_col = "job_group"
    
    #input filename
    indeed_csv_file = "data/indeed.csv"
    
    #number of pages in indeed search results to scrape
    num_pages = 1
    
    job_titles_list = ["data scientist", "machine learning engineer", "data engineer", "data analyst"]
    data = JobPostingsCollector(job_titles_list = job_titles_list, num_pages = num_pages, label_col = label_col, job_description_col = job_description_col, url_col = url_col, job_title_col = job_title_col, existing_file = indeed_csv_file)
    print(data.indeed_df.shape)


# In[10]:


class HardSkillsCollector:
    '''
    This class crawls hard skills words from various websites and then saves the results locally in a csv file.
    '''

    def __init__(self, word_col, existing_file = None):
        '''initialize variables'''
        self.word_col = word_col
        self.df_word = self._get_words()
    
    def _load_data(self, file):
        '''Read a csv file from a provided directory. Returns a pandas dataframe'''
        return pd.read_csv(file, index_col=[0])
    
    def _write_file(self, df, file):
        '''Write a pandas dataframe to a csv file in the provided directory'''        
        df.to_csv(file)
        
    def _clean_text_words(self, x):
        '''
        Removes all special characters
        
        Parameters
        ----------
        x: str. the text to be cleaned
        
        Returns
        -------
        letters_only: str. the processed text
        '''
        
        x = x.lower()
        x = re.sub(r'\([^)]*\)', "", x)
        x = re.sub("_[\_(\_[].*?[\)\]]", "", x)
        x = x.replace("_", " ")
        return x

    def _get_words(self):
        '''
        Scrapes contents from a number of websites to build the hard skills list. 
        Websites are:
        a) AnalyticsVidhya
        b) DataScienceGlossary.org
        c) Google Developer's Machine Learning Glossary
        d) Wikipedia's Glossary of Artifical Intelligence
        e) Wikipedia's list of programming languages
        
        Parameters
        ----------
        None
        
        Returns
        -------
        df_word: pandas df. dataframe of hard skills
        '''
        words = []

        #source analytics vidhya
        source = requests.get("https://www.analyticsvidhya.com/glossary-of-common-statistics-and-machine-learning-terms/#sixteen").text
        soup = BeautifulSoup(source, "lxml")
        for match in soup.find_all("td", style = "text-align: center;"):
            words.append(match.text.strip("\n").strip("\xa0").lower().strip())
        del words[0:2] #remove the heading words in the table

        #source data science glossary
        source = requests.get("http://www.datascienceglossary.org/").text
        soup = BeautifulSoup(source, "lxml")
        for match in soup.find_all("div", class_ = "col-md-3 col-sm-6"):
            match_word = match.a.text
            words.append(match_word.lower().strip())

        #source google ML glossary
        source = requests.get("https://developers.google.com/machine-learning/glossary/").text
        soup = BeautifulSoup(source, "lxml")
        for match in soup.find_all("h2", class_ = "hide-from-toc"):
            words.append(match.text.lower().strip())

        #source wikipedia - ai glossary
        source = requests.get("https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence").text
        soup = BeautifulSoup(source, "lxml")
        for match in soup.find_all("dt", class_="glossary"):
            match = match["id"]
            match_word = re.sub(r'\([^)]*\)', "", match)
            match_word = re.sub("_[\_(\_[].*?[\)\]]", "", match_word)
            match_word = match_word.replace("_", " ")
            words.append(match_word.lower().strip())

        #source wikipedia - programming languages
        source = requests.get("https://en.m.wikipedia.org/wiki/List_of_programming_languages").text
        soup = BeautifulSoup(source, "lxml")
        for match in soup.find_all("div", class_="div-col columns column-width"):
            for match_2 in match.find_all("ul"):
                for match_3 in match_2.find_all("li"):
                    text = match_3.text
                    text = re.sub(r'\([^)]*\)', "", text)
                    text = re.sub("_[\_(\_[].*?[\)\]]", "", text)
                    words.append(text.lower().strip())

        #write file
        my_dict = {'word': words}
        df_word = pd.DataFrame(my_dict)
        df_word.drop_duplicates(inplace = True)
        df_word.reset_index(drop = True, inplace = True)
        df_word[self.word_col] = df_word[self.word_col].apply(lambda x: self._clean_text_words(str(x)))
        
        return df_word
    
if __name__ == '__main__':

    #define input files
    word_col = "word"
    tc = HardSkillsCollector(word_col)
    tc._write_file(tc.df_word, "data/words_final.csv")
    


# In[ ]:




