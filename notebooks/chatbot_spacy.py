#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install spacy')


# In[ ]:


get_ipython().system('pip install langdetect')
get_ipython().system('python -m spacy download el_core_news_sm')


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from collections import Counter
import matplotlib.pyplot as plt
import nltk
import re
import random
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from langdetect import detect
# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')


# In[ ]:


df = pd.read_csv('C:/Users/Katerina/Documents/Chatbot/Customer Utterances.csv')


query_column = "Utterance"
queries = df[query_column].dropna().tolist()  # Convert to list and remove NaN values


# Load Greek SpaCy model
import spacy
nlp = spacy.load("el_core_news_sm")

  # Load stopwords
english_stopwords = set(stopwords.words('english'))
greek_stopwords = set(stopwords.words('greek'))
custom_stopwords = set(['θέλω','έχω',
    'και', 'να', 'το', 'μου', 'του', 'της', 'εγω', 'ένα', 'σε', 'με', 'για','κάνω', 'εγώ',
    'θα', 'απο', 'αυτό', 'αυτή', 'αυτο', 'ότι', 'πως', 'πολύ', 'εδώ', 'εκεί', 'είμαι','ακόμη', 'αυτές', 'στο','στις', 'στους','στα','στο','στη','δικό', 'δικού',
'δικός','εσάς', 'την', 'τα', 'είναι','μια', 'μία','έχει', 'είχαμε', 'είχα', 'έχουμε', 'έχετε','ήθελα','κάνετε', 'κάνει', 'έκανα','κάποιον', 'κάποιο', 'κάποιες','λέω', 'δω', 'μπω', 'πω', 'πείτε','τους', 'σας', 'μας','μπορείτε', 'μπορούσατε', 'μπορούσα', 'μπορεί',
'τη', 'τους', 'τo', 'η', 'οι', 'τα', 'το', 'την', 'των', 'τις', 'τον','όλων', 'όλης', 'όλο', 'όλοι', 'όλους','οποίο','ποιους', 'ποια', 'ποιο', 'ποιες','ποιον','πόση','πόσες','πόση', 'πόσα','πόσα','σ',"σε"]) 
custom_stopwords = {word.lower() for word in custom_stopwords}

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])  # punctuation
    # Process text using SpaCy
    doc = nlp(text)
    # Apply lemmatization and stopword removal (using the lemmatized form)
    text = ' '.join([token.lemma_ for token in doc if token.lemma_.lower() not in english_stopwords
                     and token.lemma_.lower() not in greek_stopwords
                     and token.lemma_.lower() not in custom_stopwords])

    return text






# In[ ]:


data = [preprocess_text(text) for text in queries]
df["processed_queries"] = data
df.to_csv("processed_queries.csv", index=False)
df


# In[ ]:


# Load Sentence-BERT model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Generate sentence embeddings
sentence_embeddings = model.encode(data)

# optimal number of clusters
inertia = []
for k in range(1, 30):  # Test 1 to 20 clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(sentence_embeddings)
    inertia.append(kmeans.inertia_)

# Elbow plot
plt.plot(range(1, 30), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# In[ ]:


from sklearn.metrics import silhouette_score

# Test different numbers of clusters
silhouette_scores = []
for k in range(2, 30):  # Start from 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(sentence_embeddings)
    score = silhouette_score(sentence_embeddings, kmeans.labels_)
    silhouette_scores.append(score)

# Plot the silhouette scores
plt.plot(range(2, 30), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.show()


# In[ ]:


# Check lemmatization for the first few queries
for query in data[:20
                  ]:  # Iterate over the first  preprocessed queries
    doc = nlp(query)  # Process the query with SpaCy
    print([token.lemma_ for token in doc])  # Print lemmatized words


# In[ ]:


num_clusters = 8

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(sentence_embeddings)

# Add cluster labels to results
results = pd.DataFrame({
    'Query': data,
    'Cluster': kmeans.labels_
})

# Automate intent labeling using top keywords
top_keywords = {}
for cluster in range(num_clusters):
    cluster_queries = results[results['Cluster'] == cluster]['Query']
    words = ' '.join(cluster_queries).split()
    top_keywords[cluster] = Counter(words).most_common(5)  # Top 5 keywords

# Print top keywords for each cluster
for cluster, keywords in top_keywords.items():
    print(f"Cluster {cluster}: {keywords}")



# In[ ]:


cluster_labels_mapping = {
    0: "--",
    1: "--",
    2: "--ν",
    3:"--",
    4: "--",
    5:"--",
    6: "--",
    7: "--"
}


# In[ ]:


# Simple CLI for testing
def predict_intent(query):

    embedding = model.encode([query])
    cluster = kmeans.predict(embedding)[0]
    return cluster_labels_mapping.get(cluster, 'unknown intent')

# Test the CLI
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    intent = predict_intent(query)
    print(f"Predicted Intent: {intent}")



# In[ ]:


# View all rows of each cluster
for cluster in range(num_clusters):
    print(f"Cluster {cluster}: {cluster_labels_mapping.get(cluster, 'Unknown')}")
    cluster_data = results[results['Cluster'] == cluster]
    print(cluster_data[['Query']])  # Display only the query column 
    print("\n" + "="*50 + "\n")


# In[ ]:





# In[ ]:




