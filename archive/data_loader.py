import pandas as pd

# This file generated & prepared data for the deprecated-text-sentiment-analysis
# Considering that the "deprecated-text-sentiment-analysis" file is now deprecated, this file is no longer in use.


"""
Cleans a string from unwanted characters
"""
def only_letters(input):
    valids = []
    for character in input:
        if character.isalpha():
            valids.append(character)
    return ''.join(valids)

"""
Cleans the word from the unwanted characters in out string data
"""
def clean_word(word):
    stripped = word.replace('.', "")
    stripped = stripped.replace('<br', "")
    stripped = stripped.replace('(', "")
    stripped = stripped.replace(')', "")
    stripped = stripped.replace('br>', "")
    stripped = stripped.replace('/', "")
    stripped = stripped.replace('\\', "")
    stripped = stripped.replace('\\\>', "")
    stripped = stripped.replace('?', "")
    stripped = stripped.replace('!', "")
    return stripped

"""
Creates a dictionary with all the words from a list of strings
returns the dictionary and the maximum number of words in a string
"""
def create_word_index(stringlist):
    counter = 0
    max_word_count = 0
    word_dict = {0: '<PADD>', 1: '<UNKNOWN>'}  # Will later be used to append a padding to the strings to match the data to the right size
    for string in stringlist:
        words = string.split()  # Splits the string into an array of words
        if len(words) > max_word_count:
            max_word_count = len(words)  # Update the maximum number of words

        for word in words:
            stripped = clean_word(word)  # Prepare word (remove '.' from the end because "word" and "word." should be treated the same
            if stripped not in list(word_dict.values()):
                word_dict[len(list(word_dict))] = stripped
            else:
                # Maybe later on also track how often a certain world is present in the data.
                # This would be best implemented here
                pass
        counter += 1
        print(counter)
        if counter % 10000 == 0:
            save_dict(word_dict, "progress_dict_" + str(counter))
    return word_dict, max_word_count

"""
Saves a dictionary as a csv file
Returns the saved dataframe
"""
def save_dict(dict, filename):
    df = pd.DataFrame.from_dict(dict, orient="index")
    df.to_csv(filename + ".csv")
    return df

"""
Flips the index and value of a given dictionary
"""
def flip_dict(dictionary):
    test = dictionary.items()
    print("length: ", len(test))
    return {v: k for k, v in dictionary.items()}

"""
Creates a list of all all strings encoded with the numbers for every word
"""
def create_input_matrix(strings, word_dict, max_words):
    output = []
    for string in strings:
        temp = []
        for word in string.split():
            try:
                x = word_dict[word]
            except:
                x = 1
            if len(temp) == max_words:  # Stop if any given string is too long
                break
            else:
                temp.append(x)
        while len(temp) < max_words:
            temp.append(0)
        output.append(temp)
    return output