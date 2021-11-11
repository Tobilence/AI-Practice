import pandas as pd
from data_loader import *
from time import time

# Read IMDB csv
df = pd.read_csv('data/imdb.csv')
df['y_output'] = (df['sentiment'] == 'positive') * 1

# Get list of 'review' column
reviewdata = list(df['review'])

# Create word index file
print("Generating word index file...")
t1 = time()
word_index_dict, max_word_count = create_word_index(reviewdata)
save_dict(word_index_dict, "complete_dict")
t2 = time()
print(f'Generating the word index file took {(t2-t1):.3} seconds')

# flipped = flip_dict(word_index_dict)
# prepared_input_data = create_input_matrix(reviewdata[:5], flipped, max_word_count)
# print(len(prepared_input_data[1]))

