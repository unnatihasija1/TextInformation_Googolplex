# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

## **Final Project : US Midterm Election – Twitter Sentiment & Topic Analysis**

* Course : CS410 Text Information Systems <br/>
* **Course Team**
    * Prateek Dhiman (Captain) : pdhiman2@illinois.edu
    * Unnati Hasija : uhasija2@illinois.edu
* Key Assets
    * Github Repo : https://github.com/pd-illinois/CS410Googolplex 
    * Main Presentation with Setup Instructions / Result of Source Code Run / Error faced with Solution : CS410_Project_USMidtermElection.pdf 
    * Project Recording on Youtube. Click the logo below to view  : 
    [![CS410](/visualization/Main_logo.png)](https://www.youtube.com/watch?v=P1Q5hqZ8f58 "CS410")

 
### **Introduction**
The social network has the capability to influence and change what people think, do, and react to everything happening around them. The social network has the capability to incept thoughts and ideas at an individual and societal level. 

Around election season, when there is a political environment around the nation, social media tools add fuel to political polarity. Through this project, we would like to identify the political sentiments and top topics that are discussed.

### **Project Overview**
We scraped Twitter data, preprocessed it and performed the following for US Midterm Election as a key area.
* Sentiment Analysis and prediction
        With Naïve Bayes Classification
        With K-Nearest Neighbor Classification
* Topic Analysis with Latent Dirichlet Allocation

### **Goals**

Key Goals for the project that we set and achieved were
* *Sentiment Analysis*  : Identify overall positive , negative and neutral sentiments towards elections and respective parties in US
* *Prediction* : Predict tweets for sentiments using Naïve Bayes and KNN. Also perform a comparison which Algorithm performs better and impact of data cleaning   techniques on precision and accuracy of predictions.
*  *Topic Analysis* : Identify top topics in tweets and predict the dominant topic and its percentage for each tweet.
*  *Visualize* : Lastly compare our findings and present them as part of our final project.

**Project Approach and Architecture:**

![Image1](/visualization/Implementation_Arch.png)

* **Contribution and credits** 
    * Step1 : Data Scrapping & Storage  : Prateek
    * Step2 : Data Preprocessing        : Prateek & Unnati
    * Step3 : Exploratory Analysis      : Prateek & Unnati
    * Step4 & 5 : Model Build & Train
        * Naive Bayes & KNN             : Unnati
        * LDA                           : Prateek
    * Step 6 : Visualizations
        * Naive Bayes & KNN             : Unnati
        * LDA                           : Prateek

#### Step 1: Data Scrapping & Storage

**PIP**
Please use pip install -r requirements.txt while you are in source folder after cloning the repository.

**Data Preparation:**

We used SNSCRAPE for capturing the tweets. We captured 20000 tweets from 1st July to 8th November (Midterm Election Day in US).
All the 20000 tweets are available in `data` Folder in `tweets_final.csv`. This is the raw data for our project that we will use. if you are interested in seeing how we scrapped the tweets , please check `source\01_Twitter_scrape.ipynb`

Note that this is a one time activity and data is already downladed and you do not need to run this script again. 
No User sensetive information was captured.
<br/>

#### Step 2: Data Preprocessing

In the Data Prepocessing areas , we aim to clean the collected tweets and use them at various stages to calculate sentiment accuracy. At this stage our goal is to check how Data Cleaning impacts Sentiment analysis.

This step is divided into three sub parts in the file `source\02_Data_Preprocess.ipynb`

a. Initial Data preperation 
    Replacing Empty Locations with Unknown
    Filtering out Non English Tweets
    
b. Generic Data Cleaning
    Lowercaseing
    Removing special characters
    Removing Whitespaces
    Removing tagged Usernames
    Removing Hashtags
    Removing RT
    Removing URLs and Http tags
    Removing Punctuations
    Removing Emojis
    Stopword Removal
    
c. NLP Specific Data Cleaning
    Lemmatization

Finally we created pickle package that will be used in exploratory analysis and Sentiment Analysis baseline (VADER)
 
#### Step 3: Exploratory Analysis and Baseline Sentiment Analysis Using VADER

Valence aware dictionary for sentiment reasoning (VADER) is a popular rule-based sentiment analyzer. It uses a list of lexical features (e.g. word) which are labeled as positive or negative according to their semantic orientation to calculate the text sentiment. Vader sentiment returns the probability of a given input sentence to be postive, negative, neutral.

Vader is optimized for social media data and can yield good results when used with data from twitter, facebook, etc.
The source file is `source\03_Exploratory_Analysis.ipynb`

*Cleaned Dataset Columns:*

    Date                    Date on which tweet was posted	
    ID                      Tweet ID
    location                Location from where tweet was posted	
    tweet                   Content of the tweet	
    num_of_likes            Number of likes on the tweet	
    num_of_retweet          Number of retweets	
    language                language of the tweet, in our case it's english	
    cleaned_tweets          cleaned tweets by removing the punctuations	
    final_cleaned_tweets    cleaned tweets using lemmatization	

 **Exploratory Data Analysis Results**

 1. What are the words/topics discussed:
 
 ![image2](https://user-images.githubusercontent.com/109382284/206315656-25ceffac-1122-4c34-b7b8-75eadc4b30da.png)

As we see here, the words seen pre and post lemmatization are almost same. We wanted to see the change lemmatization brings on overall topic modeling and sentiment analysis. since, the words are similar, we assume our accuracy on both pre and post lemmatization should be similar. 
 
 2. What are the most popular hashtags of each tweet type ?
 
 ![image3](https://user-images.githubusercontent.com/109382284/206284240-af6be292-19b1-4dd7-8e3b-0fcb0df0c3f3.png)
 
 
 For scraping our tweets, we have used the words vote, voting, elections, etc. and we see those are mostly commonly used words in the tweets and all the other words  are related to US midterm elections.
 
 3. What is the overall polarity of the tweets ?
 
 ![image4](https://user-images.githubusercontent.com/109382284/206284649-ae9d4ab4-2361-4332-bfc1-40e7f6b35605.png)

 
The above bar graph shows the three subset of data. 
    * Original 
    * Cleaned ( without lemmatization)
    * NLPCleaned ( With Lemmatization)

    **Conclusion** - We can see that positive sentiment declined while Negative sentiment were identified more in context of our research. The final data set of NLP cleaned is a fairly balanced dataset for further analysis. This also serves as our baseline dataset for prediction and topic analysis. 

#### Step 4 and 5: Train and Build Naive Bayes , KNN and LDA with Model Evaluations
 
**Topic Modeling using LDA**

In NLP , LDA is a Generative Statistical Model , which is based on a distributional hypothesis that similar topics make use of similar words and statistical mixture hypothesis that documents talk about several topics. And such a statistical distribution can be determined (Dirichlet Distribution)
LDA uses an unsupervised learning approach and is a Bayesian Version of PLSA and its parameters are regularized.

We have used Gensim library to build the corpus and dictionary for our LDA model. After multiple iterations and hyperparamterization , we concluded that at 10 topic our model performed at the best with 0.41 Coherance score. The results are discussed in the visualization section and our recorded video.


**Sentiment Analysis using Naive Bayes**

In this project, we have taken Vader Sentiment Analyzer to be our baseline for generating the labels for positive, negative and neutral. While implementing, we had also used TextBlob sentiment analyzer but found Vader sentiment analyzer to be better in distributing the data. (with TextBlob, there were more tweets classified as Neutral as compared with Vader).

For training the model, we use 75 % of the data and rest 25% for testing.

We have tried use Naive Bayes algorithm with both Count vectorizer which performs the task of tokenizing and counting and TF-IDF focuses on the frequency of words present in the corpus and also provides the importance of the words.
        
**Sentiment Analysis using K-Nearest Neighbor**
        
We also tried to use the discriminative algorithm: K-Nearest neighbor for predicting the sentiment of the same above mentioned data set. We have taken vader sentiment analyzer to be our baseline for generating the labels for positive, negative and neutral. Here also, we import the same cleaned and pre-processed data as in Naive Bayes but we have used 80% of the data for training and rest 20% for testing.
Building the feature vector for K-NN was a very time taking step. 

#### Step 6: Visualizations

**Results Comparison**
        
 *Naive Bayes using Count Vectorizer*:

![Image1](/visualization/NB_1.png)
        
 - As the value of n-grams increase, accuracy, precision, recall and F1-score decrease.
        
  *Naive Bayes using Tf-IDF*:
 
        
![image5](/visualization/NB_2.png)

        
 - Same as Naive Bayes using Count Vectorizer since the stopwords which become the common words were already removed.
        
  *K-NN*
        
![image5](/visualization/KNN.png)

        
  - Accuracy, Recall and F1 score decrease as value of k increases, Precision increases.
  - Accuracy as compared with Naive Bayes is less.

  *LDA*


  From LDAvis help ,we can visualize the top 10 Topics and how they overlap. The HTML file is available in visualization folder for you to test.
  We see the distribution of words as well the top topics and its relevance in the document.
![image6](/visualization/LDA.png)


**Conclusion and Future work**

- Accuracy improves if the data has been cleansed properly. We removed the words that are not a part of nltk.words. This helped in improving the accuracy for Naïve  Bayes from 65% to 75%. 
- For KNN, difficult to build a feature vector of huge number of tweets. 
- For sentiment analysis on tweets data, we found Naïve Bayes to show better accuracy over K-NN.
- It was more difficult to train K-NN as compared with Naïve Bayes. Hence, we conclude to use Naïve Bayes for sentiment analysis on Tweets data.
- For future work, the same model could be evaluated for different subjects other than US midterm elections by simply scraping the data on another topic from Twitter using the code provided and plugging that data for Topic Modeling and Text Categorization and could be evaluated for accuracy.
- Future work could also include modeling other text categorization techniques like SVM, Deep Learning (LSTM), etc. with word embedding techniques such as Word2vec could also be applied and compared for accuracy.



#### **REFERENCES**

https://betterprogramming.pub/how-to-scrape-tweets-with-snscrape-90124ed006af
https://github.com/cjhutto/vaderSentiment
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
https://stackoverflow.com/questions/48541801/microsoft-visual-c-14-0-is-required-get-it-with-microsoft-visual-c-build-t



