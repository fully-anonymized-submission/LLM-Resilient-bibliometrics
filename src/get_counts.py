#import Counter
from collections import Counter
import os
#Improt tqdm
import tqdm
# import pickle
import pickle

def get_counts(path):
    """ Get the counts of the words in the files in the folder.

        Parameters
            path (str): the path to the folder with the files

        Returns
            counter (Counter): a Counter object with the counts of the words
    """
    counter = Counter()
    #iterate over the files in the folder using tqdm
    for file in tqdm.tqdm(os.listdir(path)):
        #read the file
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            text = f.read()
            #split the text into words
            words = text.split()
            # for every word that appears over 5 times, increment the count
            unique_words = set(words)
            for word in unique_words:
                if words.count(word) > 5:
                    if word in counter:
                        counter[word] += 1
                    else:
                        counter[word] = 1
    return counter

if __name__ == '__main__':
    """ For each word in the corpus, we count the number of papers in which the word appears at least THRESHOLD times. """
    cwd = os.getcwd()

    ################# SETTINGS #################
    PATH_PAPER = cwd + ''
    PATH_SAVE_COUNTS = cwd + ''
    ###########################################

    # get the counts
    counts = get_counts(PATH_PAPER)
    # save the counts
    with open(PATH_SAVE_COUNTS, 'wb') as f:
        pickle.dump(counts, f)

    print(f'Counts saved to {PATH_SAVE_COUNTS}')
    print('Done')