import regex as re
import tensorflow as tf
import torch

import numpy as np
import nltk

import os
import spacy

import pandas as pd
import textacy

import string
from collections import Counter

import pickle
import tqdm
from datasets import load_dataset
#import stopwords
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
#download stopwords

nltk.download('stopwords')
nlp = spacy.load('en_core_web_lg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print the nvcc version
print('nvcc version: ', torch.version.cuda)
print('Device being used: ', device)


def get_sentences(text, min_length=5):
    """ Split the text into sentences and remove sentences under a certain length

    Parameters: 
        text (str): The text to split into sentences
        min_length (int): The minimum length of a sentence to keep

    Returns:
        list[str]: A list of sentences [sentence1, sentence2, ...]
    """
    text = text.replace('\n', ' ')
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence.split()) > min_length]
    return sentences

def split_char(text):
  """ Split the text into characters

    Parameters:
        text (str): The text to split into characters

    Returns:
        str: A string of characters
  """
  return " ".join(list(text))

def preprocess_sentences(sentences):
    """ Preprocess the sentences to be used as input for the claim model

    Parameters:
        sentences (list[str]): A list of sentences

    Returns:
        tuple: A tuple containing two numpy arrays, the first containing the sentences, the second containing the characters of the sentences
    """
    chars = [split_char(sentence) for sentence in sentences]
    # Combine chars and tokens into a numpy array
    input_data = (np.array(sentences), np.array(chars))
    return input_data

def classify_sentences(text, claim_model, min_length_sentence=5, threshold=0.5):
    """ Classify the sentences in the text as claims or not claims

    Parameters:
        text (str): The text to classify
        claim_model (tf.keras.Model): The claim model
        min_length_sentence (int): The minimum length of a sentence to keep
        threshold (float): The threshold for classifying a sentence as a claim

    Returns:
        tuple: A tuple containing two numpy arrays, the first containing the predictions, the second containing the sentences
    """

    sentences = get_sentences(text, min_length_sentence)
    input_data = preprocess_sentences(sentences)
    # Get predictions from model
    preds = claim_model.predict(input_data)
    # preds have two values, along dimension 1. The first value is the probability of the sentence being a claim, the second is the probability of it not being a claim.
    # If first value is greater than threshold, classify as claim, else classify as not claim
    preds = np.where(preds[:, 1] > threshold, 1, 0)
    
    return preds, sentences

def extract_claims(folder, save_folder, claim_model, log_errors, threshold=0.5):
    """ Extract the claims from the text in the folder and save them to a file

    Parameters:
        folder (str): The folder containing the text files
        save_folder (str): The folder to save the claims to
        claim_model (tf.keras.Model): The claim model
        log_errors (str): The file to write errors to
        threshold (float): The threshold for classifying a sentence as a claim
    """

    for filename in os.listdir(folder):
        # read in the text
        with open(folder + filename, 'r', encoding='utf-8') as f:
            text = f.read()
        # classify the sentences
        if text == '':
            continue
        # try to classify the sentences, if it throws an error, write the error to a log file
        try:
            preds, sentences = classify_sentences(text, claim_model, threshold=threshold)
        except Exception as e:
            with open(log_errors, 'a', encoding='utf-8') as f:
                f.write('Error in file ' + filename + ': ' + str(e) + '\n')
            continue
        save_path = save_folder + filename
        with open(save_path, 'w', encoding='utf-8') as f:
            for sentence, pred in zip(sentences, preds):
                if pred == 1:
                    f.write(sentence + '\n')

def extract_triplets(folder, save_folder):
    """ Extract the triplets from the text in the folder and save them to a file

    Parameters:
        folder (str): The folder containing the text files
        save_folder (str): The folder to save the triplets to
    """

    dict = {}
    for filename in os.listdir(folder):
        with open(folder + filename, 'r', encoding='utf-8') as f:
            claim = f.read()
        doc = nlp(claim)
        triplets = textacy.extract.subject_verb_object_triples(doc)
        triplets_list = []
        for triplet in triplets:
            subject, verb, object = triplet
            # subject, verb, object is a list of tokens. Make it a string of tokens
            subject = ' '.join([token.text for token in subject])
            verb = ' '.join([token.text for token in verb])
            object = ' '.join([token.text for token in object])
            triplets_list.append((subject, verb, object))
        dict[filename] = triplets_list
    # make dataframe with 2 columns: paper and triplets. The column triplets should contain a list of all triplets
    df = pd.DataFrame(list(dict.items()), columns=['paper', 'triplets'])
    # save the dataframe as a csv file
    df.to_csv(save_folder + 'triplets.csv', index=False)

def check_pos_tag(triplet, allowed_pos_tags_subject, allowed_pos_tags_object):
    """ Check if the pos tags of the subject and object are in the allowed pos tags

    Parameters:
        triplet (tuple): A triplet (subject, verb, object)
        allowed_pos_tags_subject (list[str]): A list of allowed pos tags for the subject
        allowed_pos_tags_object (list[str]): A list of allowed pos tags for the object

    Returns:
        tuple: A tuple containing a boolean indicating if the pos tags are allowed, and a string indicating if the subject or object is not allowed

    """
    subject, verb, object = triplet
    # use pos tagging on the subject, check if the pos tag is in allowed_pos_tags
    subject_pos_tags = nltk.pos_tag(nltk.word_tokenize(subject))
    object_pos_tags = nltk.pos_tag(nltk.word_tokenize(object))
    # if there is no allowed pos tag in any of the subject words
    if all([pos_tag not in allowed_pos_tags_subject for _, pos_tag in subject_pos_tags]):
        return False, 'subject'
    if all([pos_tag not in allowed_pos_tags_object for _, pos_tag in object_pos_tags]):
        return False, 'object'
    return True, None

def filter_pos_tag_triplet(subject_pos_tags, object_pos_tags, subject, verb, object, LOG_PATH, allowed_pos_tags_subject, allowed_pos_tags_object):
    """ Filter the pos tags of the subject and object

    Parameters:
        subject_pos_tags (list[tuple]): A list of tuples containing the pos tags of the subject
        object_pos_tags (list[tuple]): A list of tuples containing the pos tags of the object
        subject (str): The subject
        verb (str): The verb
        object (str): The object
        LOG_PATH (str): The path to the log file
        allowed_pos_tags_subject (list[str]): A list of allowed pos tags for the subject
        allowed_pos_tags_object (list[str]): A list of allowed pos tags for the object

    Returns:
        tuple: A tuple containing the new subject, verb and object
    """

    # maintain the pos tags in the subject and object, if the subject or object changes, write to the log file
    new_subject = ' '.join([word for word, pos_tag in subject_pos_tags if pos_tag in allowed_pos_tags_subject])
    new_object = ' '.join([word for word, pos_tag in object_pos_tags if pos_tag in allowed_pos_tags_object])
    if new_subject != subject:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write('Old subject: ' + subject + '\n' + 'New subject: ' + new_subject + '\n' + '\n')
    if new_object != object:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write('Old object: ' + object + '\n' + 'New object: ' + new_object + '\n' + '\n')
    return (new_subject, verb, new_object)

def filter_pos_tag(df, LOG_PATH, allowed_pos_tags_subject=['NN', 'NNS', 'NNP', 'NNPS'], allowed_pos_tags_object=['NN', 'NNS', 'NNP', 'NNPS', 'VBG', 'JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
    """ Filter the pos tags of the subject and object

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        LOG_PATH (str): The path to the log file
        allowed_pos_tags_subject (list[str]): A list of allowed pos tags for the subject
        allowed_pos_tags_object (list[str]): A list of allowed pos tags for the object

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
    """

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        # write allowed pos tags to the log file
        f.write('We keep the pos tags in the subject and object that are in the following lists.\n')
        f.write('Allowed POS tags for the subject: ' + str(allowed_pos_tags_subject) + '\n')
        f.write('Allowed POS tags for the object: ' + str(allowed_pos_tags_object) + '\n' + '\n')
        for index, row in df.iterrows():
            new_triplets = []
            for triplet in row['triplets']:
                subject, verb, object = triplet
                subject_pos_tags = nltk.pos_tag(nltk.word_tokenize(subject))
                object_pos_tags = nltk.pos_tag(nltk.word_tokenize(object))
                # if the subject does not have any of the allowed pos tags, remove the triplet
                if all([pos_tag not in allowed_pos_tags_subject for _, pos_tag in subject_pos_tags]):
                    # write to the log file
                    f.write('Removed triplet ' + str(triplet) + ' because the subject does not have any of the allowed POS tags.\n')
                    continue
                # if the object does not have any of the allowed pos tags, remove the triplet
                if all([pos_tag not in allowed_pos_tags_object for _, pos_tag in object_pos_tags]):
                    # write to the log file
                    f.write('Removed triplet ' + str(triplet) + ' because the object does not have any of the allowed POS tags.\n')
                    continue
                # keep only the pos tags that are in the allowed_pos_tags
                new_triplets.append(filter_pos_tag_triplet(subject_pos_tags, object_pos_tags, subject, verb, object, LOG_PATH, allowed_pos_tags_subject, allowed_pos_tags_object))
            df.at[index, 'triplets'] = new_triplets
    return df

def lower_case(df):
    """ Lower case the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets

    Returns:
        pd.DataFrame: The dataframe containing the lower cased triplets
    """
    df['triplets'] = df['triplets'].apply(lambda x: [(subject.lower(), verb.lower(), object.lower()) for subject, verb, object in x])
    return df

def filter_length(df, PATH_LOG, cutoff_length=10):
    """ Filter the triplets based on length

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        PATH_LOG (str): The path to the log file
        cutoff_length (int): The cutoff length

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
    """

    with open(PATH_LOG, 'w', encoding='utf-8') as f:
        num_removed = 0
        for index, row in df.iterrows():
            new_triplets = []
            for i in range(len(row['triplets'])):
                triplet = row['triplets'][i]
                subject, verb, object = triplet
                if len(subject.split()) > cutoff_length or len(verb.split()) > cutoff_length or len(object.split()) > cutoff_length:
                    f.write('Removed triplet ' + str(triplet) + ' because the length of the subject, verb or object exceeds the cutoff length.\n')
                    num_removed += 1
                else:
                    new_triplets.append(triplet)
            df.at[index, 'triplets'] = new_triplets
        f.write('Removed ' + str(num_removed) + ' triplets in total.\n')
    return df

def lemmatize(df, PATH_LOG):
    """ Lemmatize the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        PATH_LOG (str): The path to the log file

    Returns:
        pd.DataFrame: The dataframe containing the lemmatized triplets
    """

    with open(PATH_LOG, "a", encoding='utf-8') as f:
        f.write("Lemmatizing the triplets\n")
        lemmatizer = WordNetLemmatizer()
        triplets_lemmatized = []
        triplets = df['triplets'].values
        for triplet_list in triplets:
            new_triplet_list = []
            for triplet in triplet_list:
                subject, verb, object = triplet
                # subject, verbs and objects are lemmatized, note that they may consist of multiple words
                subject_new = ' '.join([lemmatizer.lemmatize(word) for word in subject.split()])
                object_new = ' '.join([lemmatizer.lemmatize(word) for word in object.split()])
                verb_new = ' '.join([lemmatizer.lemmatize(word) for word in verb.split()])
                # If the subject, verb or object changed, log it
                if subject_new != subject:
                    f.write(f"Subject: {subject} -> {subject_new}\n")
                if object_new != object:
                    f.write(f"Object: {object} -> {object_new}\n")
                if verb_new != verb:
                    f.write(f"Verb: {verb} -> {verb_new}\n")
                new_triplet_list.append((subject_new, verb_new, object_new))
            # append the new triplet list
            triplets_lemmatized.append(new_triplet_list)
    # update the dataframe
    df['triplets'] = triplets_lemmatized
    return df

def keep_only_text(df, PATH_LOG):
    """ Keep only the text in the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        PATH_LOG (str): The path to the log file

    Returns:
        pd.DataFrame: The dataframe containing the triplets with only text
    """
    with open(PATH_LOG, "a", encoding='utf-8') as f:
        f.write("Removing punctuation\n")
        # we want to keep letters, numbers and hyphens, but remove any other character
        to_keep = string.ascii_letters + string.digits + "-" + " "
        triplets = df['triplets'].values
        triplets_no_punctuation = []
        for triplet_list in triplets:
            new_triplet_list = []
            for triplet in triplet_list:
                subject, verb, object = triplet
                # keep only the characters that are in to_keep
                subject_new = ''.join([c for c in subject if c in to_keep])
                object_new = ''.join([c for c in object if c in to_keep])
                verb_new = ''.join([c for c in verb if c in to_keep])
                # if the subject, verb or object changed, log it
                if subject_new != subject:
                    f.write(f"Subject: {subject} -> {subject_new}\n")
                if object_new != object:
                    f.write(f"Object: {object} -> {object_new}\n")
                if verb_new != verb:
                    f.write(f"Verb: {verb} -> {verb_new}\n")
                new_triplet_list.append((subject_new, verb_new, object_new))
            # append the new triplet list
            triplets_no_punctuation.append(new_triplet_list)
    # update the dataframe
    df['triplets'] = triplets_no_punctuation
    return df

def remove_stopwords(df, PATH_LOG, redundant_verbs=['can', 'will', 'shall', 'may', 'could', 'would', 'should', 'has', 'are', 'is', 'have', 'was', 'were', 'had', 'do', 'does', 'did', 'am', 'be', 'being', 'been', 'get', 'got', 'gets', 'getting', 'gotten', 'make', 'makes', 'made', 'making', 'let', 'lets', 'letting', 'let']):
    """ Remove the stopwords from the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        PATH_LOG (str): The path to the log file
        redundant_verbs (list[str]): A list of redundant verbs

    Returns:
        pd.DataFrame: The dataframe containing the triplets with the stopwords removed
    """

    with open(PATH_LOG, "a", encoding='utf-8') as f:
        f.write("Removing stopwords\n")
        for idx, row in df.iterrows():
            triplets_col = row['triplets']
            new_triplets = []
            for triplet in triplets_col:
                subject, verb, object = triplet
                # if verb has multiple words, remove the redundant ones
                verbs = verb.split()
                if len(verbs) > 1:
                    verbs = [v for v in verbs if v not in redundant_verbs]
                    verb = ' '.join(verbs)
                # Remove stopwords from subject, verb and object, if any of them is empty, do not append the triplet
                new_subject = ' '.join([word for word in subject.split() if word not in stopwords.words('english')])
                new_object = ' '.join([word for word in object.split() if word not in stopwords.words('english')])
                if len(new_subject) > 0 and len(new_object) > 0:
                    new_triplets.append((new_subject, verb, new_object))
                if new_subject != subject:
                    f.write(f"Removed stopwords from subject: {subject} -> {new_subject}\n")
                if new_object != object:
                    f.write(f"Removed stopwords from object: {object} -> {new_object}\n")
            df.at[idx, 'triplets'] = new_triplets
    return df

def clean_up_triplets(df, PATH_LOG):
    """ Clean up the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets
        PATH_LOG (str): The path to the log file

    Returns:
        pd.DataFrame: The dataframe containing the cleaned up triplets
    """
    with open(PATH_LOG, "a", encoding='utf-8') as f:
        f.write("Cleaning up the triplets\n")
        # Remove any row that has no triplets at all
        df = df[df['triplets'].apply(lambda x: len(x) > 0)]
        # remove triplets that are empty
        num_removed = 0
        for idx, row in df.iterrows():
            triplets_col = row['triplets']
            new_triplets = []
            for triplet in triplets_col:
                subject, verb, object = triplet
                subject_words = subject.split()
                # Remove any part of the subject that has less than 3 characters
                subject = ' '.join([word for word in subject_words if len(word) > 2])
                object_words = object.split()
                # Remove any part of the object that has less than 2 characters
                object = ' '.join([word for word in object_words if len(word) > 2])
                # if the subject or object has less than 3 characters, remove the triplet
                if len(subject) < 3 or len(object) < 3 or len(verb) < 1:
                    f.write(f"Removed triplet {triplet} because the subject or object has less than 3 characters, or the verb is empty.\n")
                    num_removed += 1
                    continue
                # Replace multiple subsequent spaces with a single space
                subject = re.sub(' +', ' ', subject)
                object = re.sub(' +', ' ', object)
                verb = re.sub(' +', ' ', verb)
                new_triplets.append((subject, verb, object))
            df.at[idx, 'triplets'] = new_triplets
    return df

def get_words_from_triplets(df):
    """ Get the words from the triplets

    Parameters:
        df (pd.DataFrame): The dataframe containing the triplets

    Returns:
        list[str]: A list of words
    """
    triplets = df['triplets'].tolist()
    # flatten the list
    triplets = [item for sublist in triplets for item in sublist]
    # triplets is a list of tuples
    triplets = [list(t) for t in triplets]
    # flatten the list
    triplets = [item for sublist in triplets for item in sublist]
    # There may be multiple words, split based on space
    triplets = [t.split() for t in triplets]
    # flatten the list
    triplets = [item for sublist in triplets for item in sublist]
    # Get unique words
    words = list(set(triplets))
    # remove words that are 1 character long
    words = [w for w in words if len(w)>1]
    return words

def filter_book_corpus(book_corpus, max_length):
    """ Filter the book corpus based on length

    Parameters:
        book_corpus (list[str]): The book corpus
        max_length (int): The maximum length

    Returns:
        list[str]: The filtered book corpus
    """
    # remove all books with length over max_length
    book_corpus = [book for book in book_corpus if len(book) < max_length]
    return book_corpus

def get_corpus_frequency(corpus, words, subset=1, min_terms = 5):
    """ Get the frequency of the words in the corpus

    Parameters:
        corpus (list[str]): The corpus
        words (list[str]): The words
        subset (float): The fraction of the corpus to use
        min_terms (int): The minimum number of terms

    Returns:
        Counter: A counter containing the frequency of the words
    """

    # get fraction subset of the corpus
    corpus = corpus[:int(len(corpus)*subset)]
    document_counts = Counter()
    for i in tqdm.tqdm(range(len(corpus))):
        words_book = corpus[i].split()
        # lower case the words
        words_book = [word.lower() for word in words_book]
        # Only consider the words that are present at least min_terms times
        for word in words:
            if words_book.count(word) >= min_terms:
                document_counts[word] += 1
    return document_counts

def get_frequency_papers(PATH_FOLDER, words, min_terms = 5):
    """ Get the frequency of the words in the papers

    Parameters:
        PATH_FOLDER (str): The folder containing the papers
        words (list[str]): The words
        min_terms (int): The minimum number of terms

    Returns:
        Counter: A counter containing the frequency of the words
    """
    # get all the files in the folder
    files = os.listdir(PATH_FOLDER)
    word_freq = Counter()
    for file in tqdm.tqdm(files):
        with open(os.path.join(PATH_FOLDER, file), 'r', encoding='utf-8') as f:
            text = f.read()
            words_paper = text.split()
            # lower case the words
            words_paper = [word.lower() for word in words_paper]
            # Only consider the words that are present at least min_terms times
            for word in words:
                if words_paper.count(word) >= min_terms:
                    word_freq[word] += 1
    return word_freq

def compute_term_scores(corpus_freq, paper_freq, num_docs_corpus, num_docs_paper, min_paper_count=10):
    """ Compute the term scores

    Parameters:
        corpus_freq (Counter): The frequency of the words in the corpus
        paper_freq (Counter): The frequency of the words in the papers
        num_docs_corpus (int): The number of documents in the corpus
        num_docs_paper (int): The number of documents in the papers
        min_paper_count (int): The minimum number of papers

    Returns:
        dict: A dictionary containing the term scores
    """
    # Now make a new dictionary that is the multiplication of the two. If a key is in corpus_freq but not in paper_freq, we set the value to 0
    # if a key is in paper_freq but not in corpus_freq, and the value in paper_freq is at least 10, we set the value to infinity.
    # if a key is in paper_freq but not in corpus_freq, and the value in paper_freq is less than 10, we set the value to 0
    term_scores = {}
    for key in corpus_freq:
        if key in paper_freq and paper_freq[key] >= min_paper_count:
            corpus_score = - np.log(corpus_freq[key]/num_docs_corpus)
            paper_score = np.log(paper_freq[key]/num_docs_paper)
            term_scores[key] = corpus_score + paper_score
        else:
            term_scores[key] = - np.inf
    for key in paper_freq:
        if key not in corpus_freq:
            if paper_freq[key] >= min_paper_count:
                term_scores[key] = np.inf
            else:
                term_scores[key] = - np.inf
    # From the dictionary, remove all keys where there is punctuation, and remove all keys existing of only numbers
    term_scores_filtered = {key: value for key, value in term_scores.items() if (not any(char in string.punctuation for char in key) and not key.isdigit() and not len(key) < 2)}
    # sort term_scores from highest score to lowest
    term_scores_filtered = dict(sorted(term_scores_filtered.items(), key=lambda x: x[1], reverse=True))
    return term_scores_filtered

def filter_triplets(triplets, term_scores, threshold = 0.1):
    """ Filter the triplets based on the term scores

    Parameters:
        triplets (pd.DataFrame): The dataframe containing the triplets
        term_scores (dict): A dictionary containing the term scores
        threshold (float): The threshold

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
        list[tuple]: A list of removed triplets
    """

    # Retain the triplets that have at least one word with a score in the top threshold percent
    removed_triplets = []
    filtered_triplets = triplets.copy()

    # As a threshold, take the score of the word at the threshold percentile
    threshold_score = list(term_scores.values())[int(len(term_scores) * threshold)]
    for idx, row in tqdm.tqdm(triplets.iterrows()):
        triplet_list = row['triplets']
        kept_triplets = []
        for triplet in triplet_list:
            subject, verb, object = triplet
            object_words = object.split()
            verb_words = verb.split()
            subject_words = subject.split()
            # check if any of the words is not in the dictionary, if so, put it at negative infinity
            all_words = object_words + verb_words + subject_words
            for word in all_words:
                if word not in term_scores:
                    term_scores[word] = - np.inf
            if any([term_scores[word] >= threshold_score for word in subject_words]):
                kept_triplets.append(triplet)
            else:
                removed_triplets.append(triplet)
        filtered_triplets.at[idx, 'triplets'] = kept_triplets
    return filtered_triplets, removed_triplets

def filter_with_bookcorpus(triplets, path_book_corpus, path_papers, path_book_freq, path_paper_freq, max_length_book=30000, threshold=0.1, min_paper_count=10):
    """ Filter the triplets with the book corpus

    Parameters:
        triplets (pd.DataFrame): The dataframe containing the triplets
        path_book_corpus (str): The path to the book corpus
        path_papers (str): The path to the papers
        path_book_freq (str): The path to the book frequency
        path_paper_freq (str): The path to the paper frequency
        max_length_book (int): The maximum length of the book
        threshold (float): The threshold
        min_paper_count (int): The minimum number of papers that need to contain a word for it to be considered

    Returns:
        pd.DataFrame: The dataframe containing the filtered triplets
        list[tuple]: A list of removed triplets
    """

    # Load bookcorpus
    if os.path.isfile(path_book_corpus):
        print("Loading book corpus from file")
        book_corpus_gutenberg = pickle.load(open(path_book_corpus, "rb"))
    else:
        print("Loading book corpus from huggingface")
        book_corpus_gutenberg = load_dataset("sedthh/gutenberg_english")['train']['TEXT']
        book_corpus_gutenberg = filter_book_corpus(book_corpus_gutenberg, max_length_book)
        # save book corpus
        pickle.dump(book_corpus_gutenberg, open(path_book_corpus, "wb"))
    words_in_triplets = get_words_from_triplets(triplets)


    #if it doesn't exist, create it
    if not os.path.exists(path_book_freq):
        print("Creating book frequency")
        word_freq_book = get_corpus_frequency(book_corpus_gutenberg, words_in_triplets, subset=0.3, min_terms=5)
        with open(path_book_freq, 'wb') as f:
            pickle.dump(word_freq_book, f)
    else:
        print("Loading book frequency")
        with open(path_book_freq, 'rb') as f:
            word_freq_book = pickle.load(f)
            
    if not os.path.exists(path_paper_freq):
        print("Creating paper frequency")
        word_freq_papers = get_frequency_papers(path_papers, words_in_triplets, min_terms=5)
        with open(path_paper_freq, 'wb') as f:
            pickle.dump(word_freq_papers, f)
    else:
        print("Loading paper frequency")
        with open(path_paper_freq, 'rb') as f:
            word_freq_papers = pickle.load(f)
            
    num_docs_corpus = len(book_corpus_gutenberg)
    num_docs_paper = len(os.listdir(path_papers))
    term_scores = compute_term_scores(word_freq_book, word_freq_papers, num_docs_corpus, num_docs_paper, min_paper_count=min_paper_count)
    filtered_triplets, removed_triplets = filter_triplets(triplets, term_scores, threshold=threshold)
    return filtered_triplets, removed_triplets

def main():
    # TYPE OF RUN
    use_sample_papers = False
    use_cluster = False

    # PARAMETERS
    CUTOFF_LENGTH = 6
    THRESHOLD_CLAIMS = 0.05
    THRESHOLD_BOOKCORPUS = 0.1
    MIN_PAPER_COUNT = 10

    # PATHS
    PATH_ROOT = os.getcwd()

    if use_sample_papers:
        PATH_PROCESSED_TEXT = PATH_ROOT + "/data/processed_cited_papers/"
        PATH_CLAIMS = PATH_ROOT + "/data/claims_cited_papers/"
        PATH_TRIPLETS = PATH_ROOT + "/data/triplets_cited_papers/"
    else:
        if use_cluster:
            PATH_PROCESSED_TEXT = "/cluster/raid/data/stea/processed_arxiv_cs/"
        else:
            PATH_PROCESSED_TEXT = PATH_ROOT + "/data/processed_surveys/"
        PATH_CLAIMS = PATH_ROOT + "/data/claims_surveys/"
        PATH_TRIPLETS = PATH_ROOT + "/data/triplets_surveys/"
    
    # if folders for claims and triplets dont exist, make them
    if not os.path.exists(PATH_CLAIMS):
        os.makedirs(PATH_CLAIMS)
    if not os.path.exists(PATH_TRIPLETS):
        os.makedirs(PATH_TRIPLETS)
    
    PATH_BOOK_CORPUS = PATH_ROOT + "/data/book_corpus_gutenberg_filtered.pkl"
    PATH_SAVE_BOOK_FREQ = PATH_TRIPLETS + "word_freq_book.pkl"
    PATH_SAVE_PAPER_FREQ = PATH_TRIPLETS + "word_freq_papers.pkl"

    PATH_LOG = PATH_ROOT + '/data/logs/'
    PATH_LOG_FILTER_POS = PATH_LOG + 'log_filterpos'
    PATH_LOG_LENGTH = PATH_LOG + 'log_length'
    PATH_LOG_MAINTAIN_POS = PATH_LOG + 'log_maintainpos'
    PATH_LOG_LEMMATIZE = PATH_LOG + 'log_lemmatize.txt'
    PATH_LOG_KEEPTEXT = PATH_LOG + 'log_keeptext.txt'
    PATH_LOG_ERROR_CLAIMS = PATH_LOG + 'log_error_claims.txt'
    PATH_LOG_CLEANUP = PATH_LOG + 'log_cleanup.txt'
    PATH_LOG_STOPWORDS = PATH_LOG + 'log_stopwords.txt'

    if not os.path.exists(PATH_LOG):
        os.makedirs(PATH_LOG)

    # Clear the log files
    open(PATH_LOG_FILTER_POS, 'w').close()
    open(PATH_LOG_LENGTH, 'w').close()
    open(PATH_LOG_MAINTAIN_POS, 'w').close()
    open(PATH_LOG_LEMMATIZE, "w").close()
    open(PATH_LOG_KEEPTEXT, "w").close()
    open(PATH_LOG_ERROR_CLAIMS, "w").close()
    open(PATH_LOG_CLEANUP, "w").close()
    open(PATH_LOG_STOPWORDS, "w").close()

    # Load the claim model
    PATH_MODEL = PATH_ROOT + "/forecasting/models/claim_model"
    claim_model = tf.keras.models.load_model(PATH_MODEL)

    #if there are not yet claims in the folder, extract them
    if len(os.listdir(PATH_CLAIMS)) == 0:
        # write something to the file
        print('Extracting claims')
        extract_claims(PATH_PROCESSED_TEXT, PATH_CLAIMS, claim_model=claim_model, log_errors=PATH_LOG_ERROR_CLAIMS, threshold=THRESHOLD_CLAIMS)
    else:
        print('Loading claims from ' + PATH_CLAIMS)
    
    # check if there is the file PATH_TRIPLETS + 'triplets.csv', if not, extract the triplets
    if not os.path.exists(PATH_TRIPLETS + 'triplets.csv'):
        print('Extracting triplets')
        extract_triplets(PATH_CLAIMS, PATH_TRIPLETS)
    else:
        print('Loading triplets from ' + PATH_TRIPLETS + 'triplets.csv')

    # post-processing
    df = pd.read_csv(PATH_TRIPLETS + 'triplets.csv')
    df['triplets'] = df['triplets'].apply(lambda x: eval(x))
    # lower case everything
    df = lower_case(df)
    # Remove triplets where the length of the subject or object exceeds the cutoff length
    df = filter_length(df, PATH_LOG_LENGTH, cutoff_length=CUTOFF_LENGTH)
    # Remove stopwords from the triplets
    df = remove_stopwords(df, PATH_LOG_STOPWORDS)
    # Remove any character that is not text, except for hyphens
    df = keep_only_text(df, PATH_LOG_KEEPTEXT)
    # Remove triplets that do not have a subject or object with allowed pos tags, possible to specify the allowed pos tags
    df = filter_pos_tag(df, PATH_LOG_MAINTAIN_POS)
    # Lemmatize the triplets
    df = lemmatize(df, PATH_LOG_LEMMATIZE)
    # Clean up the triplets, remove triplets that are empty or have a subject or object with less than 3 characters
    df = clean_up_triplets(df, PATH_LOG_CLEANUP)
    # Filter the triplets with the book corpus
    df, removed_triplets = filter_with_bookcorpus(df, PATH_BOOK_CORPUS, PATH_PROCESSED_TEXT, PATH_SAVE_BOOK_FREQ, PATH_SAVE_PAPER_FREQ, threshold=THRESHOLD_BOOKCORPUS, min_paper_count=MIN_PAPER_COUNT)
    # Save the triplets
    
    df.to_csv(PATH_TRIPLETS + 'processed_triplets.csv', index=False)
    # save removed triplets
    with open(PATH_TRIPLETS + 'removed_triplets.pkl', 'wb') as f:
        pickle.dump(removed_triplets, f)
    print('Triplets are processed and saved in ' + PATH_TRIPLETS + 'processed_triplets.csv')

if __name__ == "__main__":
    main()



