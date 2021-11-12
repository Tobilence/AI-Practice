import pandas as pd

def only_letters(input):
    if isinstance(input, str):
        valids = []
        for character in input:
            if character.isalpha():
                valids.append(character)
        return ''.join(valids)
    else:
        print(input)
        return ''

df = pd.read_csv('progress_dict_20000.csv')
df.columns = ['ID', 'WORD']
print(df[:50])
df['WORD'] = df['WORD'].apply(only_letters)
df['WORD'] = df['WORD'].apply(lambda word: word.lower())
df.set_index = ['ID']
df.drop(df[df.WORD == ""].index, inplace=True)
df.drop(df[df.WORD == 'nan'].index, inplace=True)

df.to_csv('cleaned-word-dict.csv')
print(df[:50])