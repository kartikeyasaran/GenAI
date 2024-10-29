import nltk
import faiss
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim.downloader as api

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download a pre-trained word embedding model
try:
    import gensim
except ImportError:
    print("Please install gensim using 'pip install gensim'")
    exit()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download a pre-trained word embedding model
try:
    word2vec_model = api.load("word2vec-google-news-300")
except Exception as e:
    print("Error loading word2vec model:", e)
    exit()

# Check if model loaded correctly (optional)
if hasattr(word2vec_model, 'vocab'):
    print("Word2Vec model loaded successfully!")

# Preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Lowercasing
    tokens = [word.lower() for word in tokens]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

     # Convert tokens to word embeddings (using Gensim 4.0+ methods)
    embeddings = [word2vec_model.vectors[word2vec_model.key_to_index[word]] if word in word2vec_model.key_to_index else np.zeros(300)  # Or handle missing vectors differently
                   for word in tokens]

    # If there are no embeddings (all zeros), return an empty list or handle the case
    # If there are no embeddings (all zeros), return an empty list or handle the case
    if all(embedding.all() == 0 for embedding in embeddings):
        return []  # Or raise an exception

    # Average the embeddings to create a document embedding
    embedding = np.mean(embeddings, axis=0)
    return embedding

def extract_keywords(document):
    # Tokenize and use TF-IDF to find important words
    tokens = nltk.word_tokenize(document.lower())
    fdist = nltk.FreqDist(tokens)
    important_words = [word for word, freq in fdist.most_common(10)]  # Top 10 most frequent words
    return important_words


def summarize_keywords(keywords):
    # Join keywords with commas and return a summary string
    summary = ", ".join(keywords)
    return summary


def create_knowledge_base(data):
    embeddings = []

    for document in data:
        if isinstance(document, str):
            tokens = preprocess_text(document)
            embedding = np.mean([word2vec_model.wv.vectors[word2vec_model.key_to_index[word]] if word in word2vec_model.key_to_index else np.zeros(300) for word in tokens], axis=0)
            if np.all(embedding == 0):  # Check if embedding is all zeros
                continue
        else:
            embedding = np.array(document)  # Assuming numerical data

        embeddings.append(embedding)

    if not embeddings:
        print("No valid embeddings found in the data. Please check your data and word embedding model.")
        return None

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index


def answer_query(query, knowledge_base):
    if not knowledge_base:
        return "No knowledge base created (data might be empty or only contain strings)."

    query_embedding = np.array(preprocess_text(query))
    distances, indices = knowledge_base.search(np.array([query_embedding]), 1)

    if not indices:
        return "No matching documents found."

    most_relevant_document = data[indices[0][0]]

    # Extract relevant information
    keywords = extract_keywords(most_relevant_document)

    # Summarize the extracted information
    summary = summarize_keywords(keywords)

    return summary


import PyPDF2

# ... (rest of your code)

# Example usage
data = ['this is a string', 'another string', 'path/to/your/pdf.pdf']

for document in data:
    if isinstance(document, str):
        if document.endswith('.pdf'):
            with open(document, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
                data.append(text)  # Add extracted text to the data list
        else:
            # Handle other text formats or non-text data
            pass

knowledge_base = create_knowledge_base(data)
query = input('>>>>>> ')
answer = answer_query(query, knowledge_base)
print(answer)
