import nltk
import faiss
import torch
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer, AutoModel

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained RoBERTa model and tokenizer
model_name = "roberta-base"  # You can choose a different model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Lowercasing
    tokens = [word.lower() for word in tokens]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Convert tokens to word embeddings using RoBERTa
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state

    # Average the token embeddings to get a document embedding
    document_embedding = torch.mean(last_hidden_state, dim=1).squeeze().numpy()

    return document_embedding


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
            embedding = preprocess_text(document)
            if np.all(embedding == 0):
                print('all embeddings are zero')  # Check if embedding is all zeros
                continue
        else:
            embedding = np.array(document)  # Assuming numerical data

        embeddings.append(embedding)

    if not embeddings:
        print("No valid embeddings found in the data. Please check your data and word embedding model.")
        return None  # Or handle the case differently (e.g., raise an exception)

    # Ensure at least two valid embeddings before creating the index
    if len(embeddings) <= 1:
        print("Not enough valid embeddings to create knowledge base. Consider adding more documents or handling cases with a single document.")
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
data = ['Jack is running so fast on the road','The grapes are sour in taste']
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
while True :
   query = input('>>>>>> ')
   answer = answer_query(query, knowledge_base) 
   print(answer)

