# 🤖 NLU Chatbot Project (Work in Progress)

This is a Natural Language Understanding (NLU) project written in Python, where I aim to build a chatbot that can intelligently respond to customer inquiries based on their utterances.

>  **Note:** This project is still in development! I’m currently exploring two different NLP libraries: **spaCy** and **Stanza**. The code isn’t finalized yet, and I’m experimenting and learning as I go.

---

##  Project Goal

The goal of this chatbot is to understand **customer utterances** from a dataset and provide appropriate responses. The bot doesn’t use predefined intents — instead, it learns patterns and clusters them based on semantic similarity.

---

##  What I'm Doing (Step-by-Step)

1. **Preprocessing customer utterances** from a dataset
2. **Tokenizing** the text using both **spaCy** and **Stanza**
3. Creating **custom stopword lists**
4. **Mapping** certain words (e.g., synonyms or brand-specific terms) to standard forms
5. **Lemmatizing** the tokens for better generalization
6. **Vectorizing** utterances using **Sentence-BERT (SBERT)**
7. Using **KMeans Clustering** to group similar utterances
8. Assigning appropriate chatbot responses based on clusters

---

##  Approaches

I'm testing two NLP pipelines:

- `chatbot_spacy.py` – Based on spaCy
- `chatbot_stanza.py` – Based on Stanza

Each script is a work-in-progress and may evolve over time.

---



