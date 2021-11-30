import pandas as pd
from pandas import DataFrame
from pandas import Series
from text_analysis import denoise_text

class Vocab():
    _vocab = dict({0: '<UNKNOWN>'})

    def contains(self, string):
        return string in self._vocab.values()

    def append(self, string):
        next_number = len(self._vocab)
        self._vocab[next_number] = string

    def print_head(self):
        try:
            print(self._vocab[0])
            print(self._vocab[1])
            print(self._vocab[2])
            print(self._vocab[3])
            print(self._vocab[4])
        except:
            pass

    '''
    Appends every word from a list of sentences
    '''
    def append_list(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                word = denoise_text(word)
                if not self.contains(word):
                    self.append(word)

    def save(self, filename="vocab.csv"):
        keys = Series(self._vocab.keys())
        values = Series(self._vocab.values())
        df = pd.concat([keys, values], axis=1)
        df.columns = ['ID', 'WORD']
        df.set_index('ID', inplace=True)
        print(df.head())
        df.to_csv(filename)

# TODO: Read in moview reviews and create a vocabulary from that

# Read IMDB csv
df = pd.read_csv('data/imdb.csv')
df['y_output'] = (df['sentiment'] == 'positive') * 1
df['review'] = df['review'].apply(lambda text: denoise_text(text))