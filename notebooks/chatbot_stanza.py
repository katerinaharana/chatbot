#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import stanza
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords


# In[ ]:


# 🔹 Download & Load Stanza Greek Model
stanza.download('el')
nlp_stanza = stanza.Pipeline(lang='el', processors='tokenize,pos,lemma')


# In[ ]:


# 🔹 Define Custom Stopwords BEFORE Stanza Processing
custom_stopwords = set([
    "και", "να", "το", "μου", "του", "της", "εγω", "ένα", "σε", "με", "για",
    "θα", "απο", "αυτό", "αυτή", "αυτο", "ότι", "πως", "πολύ", "εδώ", "εκεί", "είμαι","ακόμη", "αυτές", "στο","στις", "στους","στα","στο","στη","δικό", "δικού"
"δικός",'εσάς', 'την', 'τα', "είναι",'μια', 'μία','έχει', 'είχαμε', 'είχα', 'έχουμε', 'έχετε','ήθελα','κάνετε', 'κάνει', 'έκανα','κάποιον', 'κάποιο', 'κάποιες','λέω', 'δω', 'μπω', 'πω', 'πείτε','τους', 'σας', 'μας','μπορείτε', 'μπορούσατε', 'μπορούσα', 'μπορεί',
'τη', 'τους', 'τo', 'η', 'οι', 'τα', 'το', 'την', 'των', 'τις', 'τον','όλων', 'όλης', 'όλο', 'όλοι', 'όλους','οποίο','ποιους', 'ποια', 'ποιο', 'ποιες','ποιον','πόση','πόσες','πόση', 'πόσα','πόσα','σ',"σε"]) 


custom_word_mapping = {
    "'τηλέφωνό": "τηλέφωνο",
    "τηλέφων": "τηλέφωνο",
    "τηλέφωνό": "τηλέφωνο",
    "τηλεφώνου": "τηλέφωνο",
    "υπολοίπου": "υπόλοιπος",
    "υποβάλω": "υπόβάλλω",
    "υποβολής" : "υποβάλλω",
    "χρωστάω" : "χρέος"
}


# In[ ]:


# 🔹 Dictionary to store word mappings (before → after lemmatization)
word_mapping = defaultdict(set)


# In[ ]:


# ✅ **Preprocessing Function (Removes Stopwords + Applies Manual Remapping)**
def preprocess_text(text):
    if pd.isnull(text):  # Handle missing values
        return ""
    
    # Convert text to lowercase & Remove punctuation
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  
    
    # Remove Stopwords before Stanza Processing
    words = text.split()  
    words = [word for word in words if word not in custom_stopwords]  # Remove stopwords
    cleaned_text = " ".join(words)  # Reconstruct sentence
    
    #  Pass Cleaned Text to Stanza for Lemmatization
    doc = nlp_stanza(cleaned_text)
    lemmatized_text = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            original_word = word.text.lower()
            lemmatized_word = word.lemma
            
            # 🔹 **Apply Custom Mapping (if exists)**
            if lemmatized_word in custom_word_mapping:
                corrected_lemma = custom_word_mapping[lemmatized_word]
            else:
                corrected_lemma = lemmatized_word  # Keep original lemma

            # Store mapping only if the lemma differs from the original word
            if original_word != corrected_lemma:
                word_mapping[corrected_lemma].add(original_word)

            lemmatized_text.append(corrected_lemma)

    return " ".join(lemmatized_text)


# In[ ]:


# 🔹 Load the dataset
df = pd.read_csv("C:/Users/Katerina/Documents/Chatbot/Customer Utterances.csv")  
df["lemmatized_queries"] = df["Utterance"].apply(preprocess_text)  


# In[ ]:


# 🔹 Convert word mappings dictionary to a DataFrame
word_mapping_df = pd.DataFrame([(lemma, list(words)) for lemma, words in word_mapping.items()], 
                               columns=["Lemmatized_Word", "Original_Words"])


# In[ ]:


#  Sentence Embeddings with Sentence-BERT
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(df["lemmatized_queries"].tolist())


# In[ ]:


#  K-Means Clustering
def optimal_clusters(embeddings, max_clusters=15):
    distortions = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters+1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        # Calculate Elbow Method distortion score
        distortions.append(kmeans.inertia_)

        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    return cluster_range, distortions, silhouette_scores

# Find the optimal number of clusters using Elbow & Silhouette methods
cluster_range, distortions, silhouette_scores = optimal_clusters(embeddings)

#  Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, distortions, marker="o", linestyle="-")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion (Inertia)")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

#  Plot Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, marker="o", linestyle="-", color="red")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Analysis")
plt.show()


# In[ ]:


#  Perform final K-Means clustering using the best number of clusters
optimal_k = 7 #cluster_range[silhouette_scores.index(max(silhouette_scores))]  # Best cluster count
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = final_kmeans.fit_predict(embeddings)

#  Function to Predict Cluster for New Input
def predict_cluster(user_input):
    processed_input = preprocess_text(user_input)
    input_embedding = model.encode([processed_input])
    cluster = final_kmeans.predict(input_embedding)[0]
    print(f"🔹 The input belongs to Cluster {cluster}")
    return cluster


# In[ ]:


# Function to Find Top 5 Most Frequent Words in Each Cluster
def get_top_words_per_cluster():
    cluster_word_counts = defaultdict(Counter)

    for _, row in df.iterrows():
        words = row["lemmatized_queries"].split()
        cluster_word_counts[row["Cluster"]].update(words)

    top_words = {cluster: word_counts.most_common(5) for cluster, word_counts in cluster_word_counts.items()}
    
    # Convert to DataFrame for better visualization
    top_words_df = pd.DataFrame([
        {"Cluster": cluster, "Top_Words": [word for word, _ in words]} for cluster, words in top_words.items()
    ])
    
    return top_words_df


# In[ ]:


# 🔹 Save processed data and mappings
df.to_csv("lemmatized_queries_with_clusters.csv", index=False)
word_mapping_df.to_csv("lemmatization_mapping_stanza.csv", index=False)
top_words_df = get_top_words_per_cluster()
top_words_df.to_csv("top_words_per_cluster.csv", index=False)


# In[ ]:


# 🔹 Display the mapping DataFrame
display(word_mapping_df)


# In[ ]:


display(top_words_df)


# In[ ]:


# 🔹 Show processed DataFrame
df.head()



# In[ ]:


# **Simple CLI for Testing**
if __name__ == "__main__":
    while True:
        query = input("\n💬 Enter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("👋 Exiting... Have a great day!")
            break
        intent = predict_intent(query)
        print(f"🔹 Predicted Intent: {intent}")


# In[ ]:





# In[ ]:




