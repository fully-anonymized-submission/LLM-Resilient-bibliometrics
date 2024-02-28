import nltk
import os
import numpy as np
import tensorflow as tf

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


def main():
    ###################################   SETTINGS  ###################################################
    THRESHOLD_CLAIMS = 0.05
    PATH_ROOT = os.getcwd()

    ################################## FILL IN THE PATHS ###############################################
    PATH_PROCESSED_TEXT = PATH_ROOT + ''
    PATH_CLAIMS = PATH_ROOT + ''
    ####################################################################################################
    
    # if folders for claims and triplets dont exist, make them
    if not os.path.exists(PATH_CLAIMS):
        os.makedirs(PATH_CLAIMS)

    PATH_LOG = PATH_ROOT + '/data/logs/'
    PATH_LOG_ERROR_CLAIMS = PATH_LOG + 'log_error_claims.txt'

    if not os.path.exists(PATH_LOG):
        os.makedirs(PATH_LOG)

    open(PATH_LOG_ERROR_CLAIMS, "w").close()

    # Load the claim model
    PATH_MODEL = PATH_ROOT + "/forecasting/models/claim_model"
    claim_model = tf.keras.models.load_model(PATH_MODEL)

    #if there are not yet claims in the folder, extract them
    if len(os.listdir(PATH_CLAIMS)) == 0:
        # write something to the file
        print('Extracting claims')
        extract_claims(PATH_PROCESSED_TEXT, PATH_CLAIMS, claim_model=claim_model, log_errors=PATH_LOG_ERROR_CLAIMS, threshold=THRESHOLD_CLAIMS)
        print('Claims are extracted')

    else:
        print('Claims are already extracted')


    
if __name__ == "__main__":
    main()