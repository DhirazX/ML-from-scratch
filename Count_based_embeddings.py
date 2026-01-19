import numpy as np

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

# create a vocabulary i.e find all the unique words in the sentences
vocab = set() 

for sentence in sentences:
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
window_size = 4     # two words left and right

word_to_idx = {}
for i, word in enumerate(vocab):
    word_to_idx[word] = i


for sentence in sentences:
    words = sentence.split()
    for i,word in enumerate(words):
        word_id  = word_to_idx[word]

        start = max(0, i- window_size);
        end = min(len(words), i + window_size + 1)

        for j in range(start, end):
            if i!= j:
                context_id = word_to_idx[words[j]]
                co_occur[word_id, context_id] += 1

print(word_to_idx)
print(co_occur[2])

print(co_occur[word_to_idx['the']][word_to_idx['cat']])























