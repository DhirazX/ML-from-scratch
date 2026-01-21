import numpy as np
import re
import numpy as np
from sklearn.datasets import fetch_20newsgroups

# example sentence
sentences = [
    'the cat sat on the mat',
    'the dog sat on the log',
    'the cat and the dog are friends',
    'cat and dog play together',
    'the cat loves the dog',
    'dog and cat sleep on the mat',
    'the mat is soft and comfortable',
    'the log is hard and rough'
]


data = fetch_20newsgroups()

print("Number of documents:", len(data.data))
# print("Categories:", data.target_names)
# print("\nFirst document preview:")
# print(data.data[0:100])


# Function to clean the dataset
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove email headers, URLs, email addresses
    text = re.sub(r'\S+@\S+', '', text)  # emails
    text = re.sub(r'http\S+', '', text)  # URLs
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Keep only letters and spaces (removes punctuation, special chars)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Clean all documents
subset_size = 2000 # limite to trim the dataset
cleaned_docs = [clean_text(doc) for doc in data.data[:subset_size]]

# print("Original:", data.data[:1])
print("\nCleaned:", cleaned_docs[:1])

# create a vocabulary i.e find all the unique words in the sentences
vocab = set() 

for sentence in cleaned_docs:
    words = sentence.split()
    vocab.update(words)

vocab = sorted(list(vocab))

# print(vocab)

''' Create a matrix of size nxn where n = vocabulary count 
    The rows and columns are the words in the vocab
    The value of the column is determined by the count of times they appear around the 
    row-word in the context_window '''

# create an empty matrix
n = len(vocab)
co_occur = np.zeros((n,n))


# define context window size
window_size = 8     # two words left and right

word_to_idx = {}
for i, word in enumerate(vocab):
    word_to_idx[word] = i


for sentence in cleaned_docs:
    words = sentence.split()
    for i,word in enumerate(words):
        word_id  = word_to_idx[word]

        start = max(0, i- window_size);
        end = min(len(words), i + window_size + 1)

        for j in range(start, end):
            if i!= j:
                context_id = word_to_idx[words[j]]
                co_occur[word_id, context_id] += 1

# print(word_to_idx)
# print(co_occur[2])

print("RESULT: ", co_occur[word_to_idx['car']][word_to_idx['history']])






















