from helpers import *
import fitz

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import os

from fastcoref import spacy_component
from pathlib import Path
import pandas as pd

from abbreviations import schwartz_hearst
from typing import Union
import re

import time
import tqdm
import nltk
nltk.download('punkt')

import spacy
from spacy.tokens import Doc
from typing import List
import random

# Set up devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device being used: ', device, flush=True)
spacy.prefer_gpu() 
print('Device being used for spacy: ', spacy.prefer_gpu(), flush=True)

# Load spacy pipeline
nlp = spacy.load('en_core_web_lg', exclude=['parser', 'ner', 'lemmatizer', 'textcat'])
nlp.add_pipe("fastcoref")

class LanguageDetector:

    """Detect the language of a text with a LLM"""

    language_model_name = "papluca/xlm-roberta-base-language-detection"

    def __init__(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.language_model_name
        )
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def get_language(self, text: str) -> list[dict]:
        """Get the language of a text

        Parameters:
            text (int): text to analyze

        Return:
            language_score list[{'label': str, 'score': int}]: list of dict with the language and the score
        """

        # If the input is too long can return an error
        try:
            return self.classifier(text[:1300])
        except Exception:
            try:
                return self.classifier(text[:514])
            except Exception:
                return [{"label": "err"}]

def get_text_files_in_dir(path):
    """ Get all the .txt files in a directory
    
    Parameters:
        path (str): path to the directory
        
        Return:
        texts (list[str]): list of texts
        text_file_names (list[str]): list of names of the text files  
    """
    texts = []
    text_file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                text_file_names.append(file)
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    return texts, text_file_names

def get_text_from_pdf(path: Union[str, Path]) -> str:
    """Extract text from a pdf file

    Parameters:
        path (str|Path): path to the pdf file

    Return:
        text (str): text extracted from the pdf file
    """
    try:
        pdf_file = fitz.open(path)
    except Exception as e:
        print(f'Error opening {path}: {e}')
        return ''
    txt = "".join([page.get_text() for page in pdf_file])
    pdf_file.close()
    return txt

def keep_categories(PATH_RAW_FILES, PATH_METADATA, categories = ['cs.ET', 'quant-ph', 'cs.CR','cs.CV', 'physics.pop-ph'], remove_files=False, inverse=False):
    """Keep only the files with the allowed categories

    Parameters:
        PATH_RAW_FILES (str): path to the directory with the raw files
        PATH_METADATA (str): path to the directory with the metadata
        categories (list[str]): list of allowed categories
        remove_files (bool): whether to remove the files that are not in the allowed categories
        inverse (bool): if True, we remove the files that are in the parameter categories

    Return:
        kept_files (list[str]): list of the kept files
    """

    if os.path.exists(PATH_METADATA + '/arxiv_metadata.json'):
        # convert json to pickle for faster loading
        df = pd.read_json(PATH_METADATA + '/arxiv_metadata.json', lines=True)
        df.to_pickle(PATH_METADATA + '/arxiv_metadata.pickle')
    else:
        df = pd.read_pickle(PATH_METADATA + '/arxiv_metadata.pickle')
    if inverse:
        df_filtered = df[df['categories'].apply(lambda x: all([c not in x.split() for c in categories]))]
    else:
        df_filtered = df[df['categories'].apply(lambda x: any([c in x.split() for c in categories]))]
    print('Percentage of articles from metadata with required categories: {:.2f}%'.format(100*len(df_filtered)/len(df)), flush=True)
    kept_files = df_filtered["id"].to_list()
    if remove_files:
        num_removed = 0
        num_kept = 0
        # Now iterate over the raw files
        for subdir, dirs, files in os.walk(PATH_RAW_FILES):
            for filename in files:
                # remove the suffix .pdf and the version
                filename_reduced = filename[:-6]
                if filename_reduced not in kept_files:
                    path_for_removal = os.path.join(subdir, filename)
                    os.remove(path_for_removal)
                    num_removed += 1
                else:
                    num_kept += 1
        print('Percentage removed: ', num_removed/(num_removed + num_kept))
    return kept_files

def get_all_pdf_from_dir(path: str, kept_files=None, subset=None) -> list[Path]:
    """Get all the pdf files from a directory

    Parameters:
        path (str): path to the directory

    Return:
        pdf_list (list[Path]): list of Path to the pdf files
    """
    # also search subdirectories
    pdf_list = []
    for root, dirs, files in os.walk(path):
        random_files = random.sample(files, subset) if subset is not None else files
        for file in random_files:
            if kept_files is None:
                if file.endswith(".pdf"):
                    pdf_list.append(Path(root) / file)
            else:
                if file.endswith(".pdf") and file[:-6] in kept_files:
                    pdf_list.append(Path(root) / file)
    print('Subset: ', subset, flush=True)

    return pdf_list

def remove_header_references(text: str) -> str:
    """Remove the header and the references from a text

    Parameters:
        text (str): text to clean

    Return:
        text (str): cleaned text
    """
    txt_lower = text.lower()
    abstract_pos = txt_lower.find("abstract")
    introduction_pos = txt_lower.find("introduction")

    if introduction_pos != -1 and abstract_pos != -1:
        abstract_pos = min(abstract_pos, introduction_pos)
    else:
        abstract_pos = max(abstract_pos, introduction_pos)

    if abstract_pos == -1:
        # If not foud remove fixed number of characters to remove part of the header
        abstract_pos = 100

    references_pos = txt_lower.rfind("reference")
    acknowledgements_pos = txt_lower.rfind("acknowledgement")
    if (
        acknowledgements_pos != -1
        and acknowledgements_pos < references_pos
        and acknowledgements_pos > len(text) / 2
    ):
        references_pos = acknowledgements_pos
    if references_pos == -1:
        references_pos = len(text)

    return text[abstract_pos:references_pos]

def process_pdfs(save_folder, pdf_list: list[Path]):
    """Process the pdfs

    Parameters:
        save_folder (str): folder to save the processed texts
        pdf_list (list[Path]): list of Path to the pdf files
        
    Return:
        texts (list[str]): list of texts
    """
    # load the language detector and the keyword extractor
    language_detector = LanguageDetector()
    texts = []
    for pdf in pdf_list:
        txt = get_text_from_pdf(pdf)
        if txt == '':
            continue
        # process only english papers or papers with no language detected 
        if language_detector.get_language(txt)[0]["label"] not in ["en", "err"]:
            continue
        txt = remove_header_references(txt)
        pdf_name = pdf.name[:-4]
        # split text into sentences
        txt = txt.replace('\n', ' ')
        sentences = nltk.sent_tokenize(txt)
        
        # if a sentence ends with 'et al.', then merge it with the next sentence (this is a common failure mode of the sentence tokenizer)
        for i in range(len(sentences) - 1):
            if i + 1 < len(sentences):
                num_replacements = 0
                while sentences[i].endswith("et al."):
                    sentences[i] = sentences[i] + " " + sentences[i + 1 + num_replacements]
                    num_replacements += 1
                # remove the merged sentences
                sentences = sentences[:i + 1] + sentences[i + 1 + num_replacements:]
                
        # Save the text in a txt file where each sentence is on a new line
        with open(os.path.join(save_folder, pdf_name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))

        texts.append("\n".join(sentences))
    return texts

def keep_newest_versions(folder_name):
    """Keep only the newest version of each paper, remove the older versions

    Parameters:
        folder_name (str): path to the directory with the pdf files

    Return:
        None
    """
    files_to_keep = set()
    num_files = 0
    version_dict = {}

    # iterate over the files and keep the newest version
    for subdir, _, files in os.walk(folder_name):
        for filename in files:
            num_files += 1
            filename_reduced = filename.split('v')[0]
            version = int(filename.split('v')[-1].split('.')[0])
            # if the filename is in the dictionary, check whether the version is higher
            if filename_reduced in version_dict:
                if version > version_dict[filename_reduced]:
                    version_dict[filename_reduced] = version
            else:
                version_dict[filename_reduced] = version

    for filename_reduced, version in version_dict.items():
        files_to_keep.add(filename_reduced + 'v' + str(version) + '.pdf')
    num_files_to_keep = len(files_to_keep)
    print('Percentage of files removed due to being older versions: ', (num_files - num_files_to_keep)/num_files, '%')
    # remove all files that are not in files_to_keep
    for subdir, _, files in os.walk(folder_name):
        for filename in files:
            if filename not in files_to_keep:
                path_for_removal = os.path.join(subdir, filename)
                os.remove(path_for_removal)
        
def remove_citations(texts, log_file):
    """Remove citations through a rule based heuristic

    Parameters:
        texts (list[str]): list of texts
        log_file (str): path to the log file
    Return:
        new_texts (list[str]): list of texts with the citations removed

    """
    new_texts = []
    for text in texts:
        new_text = ''
        line_count = 0
        for line in text.split('\n'):
            # Brackets with only a number inside are removed
            # Brackets with a year inside are removed
            # Brackets with a number inside and other text, e.g. [llm2], are not removed
            re_expression = '\[[0-9]{4}[a-zA-Z0-9 .,!/\-"\']*\]|\[[0-9]+\]|\[[a-zA-Z0-9 .,!/\-"\']*[0-9]{4}\]|\([a-zA-Z0-9 .,!/\-"\']*[0-9]{4}\)|\([0-9]{4}[a-zA-Z0-9 .,!/\-"\']*\)|\([0-9]+\)'
            if re.search(re_expression, line):
                # get starting and ending position of citation. If there are multiple citations in one line, store starting and ending position of each in a list
                new_line = re.sub(re_expression, '', line)
                start_pos, end_pos = [], []
                for match in re.finditer(re_expression, line):
                    start_pos.append(match.start())
                    end_pos.append(match.end())
                write_log(line, new_line, line_count, start_pos, end_pos, 'Removing citations', log_file)
            else:
                new_line = line
            line_count += 1
            new_text += new_line + '\n'
        new_texts.append(new_text)
    return new_texts

# ------------------- EXPAND ABREVIATIONS -------------------

def expand_abbreviations(texts, log_file):
    """Expand the abbreviations using the Schwartz-Hearst algorithm

    Parameters:
        texts (list[str]): list of texts
        log_file (str): path to the log file

    Return:
        new_texts (list[str]): list of texts with the abbreviations expanded
        pairs (dict): dictionary with the abbreviations as keys and the definitions as values
    """

    new_texts = []
    errors_with_abbreviations = set()
    for text in texts:
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=text)
        # Add the fully lowercased versions of the abbreviations as keys
        pairs_copy = pairs.copy()
        for abbrev, definition in pairs_copy.items():
            if abbrev.lower() != abbrev:
                pairs[abbrev.lower()] = definition
        # iterate over the lines in the text file and replace the abbreviations
        # split by \n to get the lines
  
        sentences = text.split('\n')
        new_sentences = []
        for i, sentence in enumerate(sentences):
            old_sentence = sentence
            start_pos, end_pos = [], []
            replacements = []
            for abbrev, definition in pairs.items():
                # check whether the abbreviation is in the sentence
                if abbrev in sentence:
                    # we have to make sure that the abbreviation is not inside a word, e.g. "in" in "within". It is allowed to have punctuation before and after the abbreviation, e.g. AI, or AI.
                    # We add a "try" since the abbreviation might contain a backslash, which would cause an error. If there is an error, we skip the abbreviation
                    try:
                        for m in re.finditer(abbrev, old_sentence):
                            # check whether there is a letter before and after the abbreviation
                            if m.start() > 0:
                                if sentence[m.start()-1].isalpha():
                                    continue
                            if m.end() < len(sentence):
                                if sentence[m.end()].isalpha():
                                    continue
                            replacements.append(((m.start(), m.end()), definition))
                    except:
                        errors_with_abbreviations.add(abbrev)
                        continue
            # Now we want to make sure that the replacements do not overlap. We do this by sorting the replacements by their start index and then iterating over them and only keeping the first replacement that does not overlap with the previous replacements
            replacements = sorted(replacements, key=lambda x: x[0][0])
            replacements_to_keep = []
            for replacement in replacements:
                if len(replacements_to_keep) == 0:
                    replacements_to_keep.append(replacement)
                else:
                    # check whether the replacement overlaps with the previous replacements
                    overlap = False
                    for replacement_to_keep in replacements_to_keep:
                        if replacement[0][0] <= replacement_to_keep[0][1]:
                            overlap = True
                            break
                    if not overlap:
                        replacements_to_keep.append(replacement)
            # Now we can replace the abbreviations with their definitions
            sorted_replacements_to_keep = sorted(replacements_to_keep, key=lambda x: x[0][0], reverse=True)
            for replacement in sorted_replacements_to_keep:
                sentence = sentence[:replacement[0][0]] + replacement[1] + sentence[replacement[0][1]:]
                start_pos.append(replacement[0][0])
                end_pos.append(replacement[0][1])
            new_sentences.append(sentence)
            if (len(replacements_to_keep) > 0):
                write_log(old_sentence, sentence, i, start_pos, end_pos, 'Abbreviation replacement', log_file)
        # Get new_text by joining the sentences
        new_text = '\n'.join(new_sentences)
        new_texts.append(new_text)
    # print the abbreviations that caused errors
    print('Errors with abbreviations: ', errors_with_abbreviations)
    return new_texts, pairs

# ------------------- COREFERENCE RESOLUTION -------------------

def get_span_noun_indices(doc, cluster) -> List[int]:
    """Get the indices of the spans that contain a noun

    Parameters:
        doc (Doc): spacy document
        cluster (list[tuple]): list of tuples with the start and end position of the spans

    Return:
        span_noun_indices (list[int]): list of indices of the spans that contain a noun
    """

    spans = [doc.text[span[0]:span[1]+1] for span in cluster]
    # We now want to know which tokens are in the spans and whether they are nouns
    span_noun_indices = []
    for idx, span in enumerate(spans):
        has_noun = False
        for token in doc:
            if token.text in span and token.pos_ in ['NOUN', 'PROPN']:
                has_noun = True
                break
        if has_noun:
            span_noun_indices.append(idx)
    return span_noun_indices

def is_containing_other_spans(span, all_spans):
    """Check whether a span is containing other spans

    Parameters:
        span (tuple): tuple with the start and end position of the span
        all_spans (list[tuple]): list of tuples with the start and end position of the spans

    Return:
        bool: whether the span is containing other spans
    """
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

def get_cluster_head(doc: Doc, cluster, noun_indices):
    """Get the head of the cluster

    Parameters:
        doc (Doc): spacy document
        cluster (list[tuple]): list of tuples with the start and end position of the spans
        noun_indices (list[int]): list of indices of the spans that contain a noun

    Return:
        head_span (str): head of the cluster
        head_start_end (tuple): tuple with the start and end position of the head
    """
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc.text[head_start:head_end+1]
    return head_span, (head_start, head_end)

def replace_corefs(doc, PATH_LOG, clusters):
    """Replace the coreferences in the text

    Parameters:
        doc (Doc): spacy document
        PATH_LOG (str): path to the log file
        clusters (list[list[tuple]]): list of clusters, where each cluster is a list of tuples with the start and end position of the spans

    Return:
        new_text (str): text with the coreferences replaced
    """

    all_spans = [span for cluster in clusters for span in cluster]
    #initialize new text being equal to old text
    new_text = doc.text
    start_positions = []
    end_positions = []
    all_replacements = []
    for cluster in clusters:
        noun_indices = get_span_noun_indices(doc, cluster)
        if len(noun_indices) > 0:
            mention_span, mention = get_cluster_head(doc, cluster, noun_indices)
            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    # Execute the replacement
                    start_pos, end_pos = coref
                    # Replace the coref
                    start_positions.append(coref[0])
                    end_positions.append(coref[1])
                    # Store the replacement in a way that we can do it later
                    all_replacements.append((coref, mention_span))

    # Now do the replacements, take into account that the positions of the replacements change
    for idx, replacement in enumerate(all_replacements):
        start_pos, end_pos = start_positions[idx], end_positions[idx]
        _, mention_span = replacement

        begin_found, end_found = False, False
        tracker_begin, tracker_end = start_pos, end_pos
        while not (begin_found and end_found):
            if not begin_found:
                if new_text[tracker_begin] == '\n':
                    begin_found = True
                else:
                    if tracker_begin == 0:
                        begin_found = True
                    else:
                        tracker_begin -= 1
            if not end_found:
                if new_text[tracker_end] == '\n':
                    end_found = True
                else:
                    if tracker_end == len(new_text)-1:
                        end_found = True
                    else:
                        tracker_end += 1
        sentence = new_text[tracker_begin+1:tracker_end]
        start_pos_in_sentence = start_pos - tracker_begin - 1
        end_pos_in_sentence = end_pos - tracker_begin - 1
        mention_span = mention_span.lower()
        mention_span = mention_span.replace('.', '')
        mention_span = mention_span.replace(',', '')
        if new_text[start_positions[idx]-1] != ' ':
            if mention_span[0] != ' ':
                mention_span = ' ' + mention_span
        # if there is no space after, we add one. Be sure that we are not adding a space at the end of the text, this would cause an error as we would be out of range
        try:
            if end_positions[idx] < len(new_text)-1 and new_text[end_positions[idx]+1] != ' ':
                if mention_span[-1] != ' ':
                    mention_span = mention_span + ' '
        except:
            print('Error with end_positions: ', end_positions[idx], len(new_text))

        new_text = new_text[:start_positions[idx]] + mention_span + new_text[end_positions[idx]+1:]
        new_sentence = sentence[:start_pos_in_sentence] + mention_span + sentence[end_pos_in_sentence+1:]

        # write log
        write_log(sentence, new_sentence, 'Unknown', [start_pos_in_sentence], [end_pos_in_sentence], 'Coreference resolution', PATH_LOG)
        # Adapt the positions of the corefs, go over range idx until end
        for i in range(idx, len(all_replacements)):
            if start_positions[i] > start_positions[idx] and end_positions[i] > end_positions[idx]:
                # adapt start_position and end_position
                start_positions[i] = start_positions[i] - (end_positions[idx] - start_positions[idx] + 1) + len(mention_span)
                end_positions[i] = end_positions[i] - (end_positions[idx] - start_positions[idx] + 1) + len(mention_span)
                
    return new_text



def coreference_resolution(texts, PATH_LOG, batch_size=50000):
    """Resolve the coreferences in the texts using the fastcoref library

    Parameters:
        texts (list[str]): list of texts
        PATH_LOG (str): path to the log file
        batch_size (int): size of the batches to process the texts

    Return:
        new_texts (list[str]): list of texts with the coreferences resolved
    """

    new_texts = []
    # use tqdm to show progress bar
    for text in tqdm.tqdm(texts):
        # split up the text in batches of 200000 characters, split by \n
        if len(text) > batch_size:
            new_text = ''
            # we want to do coreference resolution on the text in batches of around 200000 characters, where we split by \n
            # we want to make sure that we do not split a sentence in half
            while len(text) > 50000:
                # find the position of the last \n before 200000 characters
                split_pos = text[:50000].rfind('\n')
                doc = nlp(text[:split_pos])
                clusters = doc._.coref_clusters
                new_text += replace_corefs(nlp(text[:split_pos]), PATH_LOG, clusters)
                text = text[split_pos:]
            doc = nlp(text)
            clusters = doc._.coref_clusters
            new_text += replace_corefs(doc, PATH_LOG, clusters)
            new_sentences = new_text
        else:
            doc = nlp(text)
            clusters = doc._.coref_clusters
            new_sentences = replace_corefs(doc, PATH_LOG, clusters)
    new_texts.append(new_sentences)
    # clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return new_texts

def fix_line_breaks(texts, PATH_LOG):
    """Fix line breaks in the texts

    Parameters:
        texts (list[str]): list of texts
        PATH_LOG (str): path to the log file

    Return:
        new_texts (list[str]): list of texts with the line breaks fixed
    """
    new_texts = []
    for idx, text in enumerate(texts):
        new_text = ''
        for idx, line in enumerate(text.split('\n')):
            # We start by fixing structures such as "beau- tiful" and "beau- tifully" to "beautiful" and "beautifully"
            # We also want to fix structures such as "beau-  tiful" to "beautiful", or "beau-   tiful" to "beautiful". 
            start_positions = [m.start() for m in re.finditer(r'(\w)-\s+(\w)', line)]
            end_positions = [m.end() for m in re.finditer(r'(\w)-\s+(\w)', line)]
            new_line = re.sub(r'(\w)-\s+(\w)', r'\1\2', line)
            # write log
            if len(start_positions) > 0:
                write_log(line, new_line, idx, start_positions, end_positions, 'Fixing line breaks', PATH_LOG)
            new_text += new_line + '\n'
        new_texts.append(new_text)
    return new_texts


# ------------------- MAIN -------------------

def main():
    """Process all the pdf files in the directory and extract the keywords with the different extractors"""
    ###################################   SETTINGS  ###################################################
    inverse = False # put to true if you want to remove categories instead of keeping them
    subset = None # if you want to process only a subset of the pdfs, put the number here
    filter_categories = True # if you want to filter based on the categories, put to True

    use_sample_papers = False
    use_coref_resolution = False # whether to use coreference resolution
    use_cluster = False # whether to run on a cluster
    convert_pdf_to_text = True # whether to convert pdf to text or to use pre-saved text files
    filter_by_version = False # whether to keep only the newest version of each paper
    categories = ['cs.AI', 'cs.CL', 'cs.LG'] # categories to keep (if inverse is False) or to remove (if inverse is True)
    
    # please run from the root directory of the project
    PATH_ROOT = os.getcwd()
    print('Current working directory: ', PATH_ROOT)
    ####################################################################################################

    if use_cluster:
        PATH_METADATA = '/cluster/raid/data/stea/metadata'
    else:
        PATH_METADATA = PATH_ROOT + '/data/metadata'
    if use_sample_papers:
        PATH_RAW_PDF = PATH_ROOT + '/data/cited_papers/'
        PATH_SAVE_TEXT = PATH_ROOT + '/data/processed_cited_papers/'
    else:
        if not use_cluster:
            PATH_RAW_PDF = PATH_ROOT + '/data/arxiv_subsample_1000/'
            PATH_SAVE_TEXT = PATH_ROOT + '/data/processed_arxiv_subsample_1000_excludecsquant/'
        else:
            PATH_RAW_PDF = '/cluster/raid/data/stea/arxiv/2312'
            PATH_SAVE_TEXT = '/cluster/raid/data/stea/processed_arxiv_cs_coref/'
        
    PATH_LOG = PATH_ROOT + '/data/logs/'
    # check if folders exist, otherwise create them
    if not os.path.exists(PATH_LOG):
        print('Creating log folder...')
        os.makedirs(PATH_LOG)

    if not os.path.exists(PATH_SAVE_TEXT):
        print('Creating folder to save texts... ')
        os.makedirs(PATH_SAVE_TEXT)

    PATH_LOG_CITATIONS = PATH_LOG + 'log_citations.txt'
    PATH_LOG_ABBREVIATIONS = PATH_LOG + 'log_abbreviations.txt'
    PATH_LOG_COREFERENCE = PATH_LOG + 'log_coreference.txt'
    PATH_LOG_LINE_BREAKS = PATH_LOG + 'log_line_breaks.txt'
    # Clear the log files
    open(PATH_LOG_CITATIONS, 'w').close()
    open(PATH_LOG_ABBREVIATIONS, 'w').close()
    open(PATH_LOG_COREFERENCE, 'w').close()
    open(PATH_LOG_LINE_BREAKS, 'w').close()
    
    start_time_overall = time.time()
    if convert_pdf_to_text:
        # ------------------- KEEP NEWEST PAPER VERSION -------------------
        if not use_sample_papers:
            if filter_by_version:
                print('Keeping newest paper version...', flush=True)
                keep_newest_versions(PATH_RAW_PDF)
            # ------------------- KEEP ALLOWED CATEGORIES AND GET PDFS-------------------
                
            if filter_categories:
                print('Getting list of allowed categories...', flush=True)
                kept_categories = keep_categories(PATH_RAW_PDF, PATH_METADATA, categories, remove_files=False, inverse=inverse)

            else:
                kept_categories = None

            print('Getting pdfs...', flush=True)
            pdf_list = get_all_pdf_from_dir(PATH_RAW_PDF, kept_categories, subset=subset)

        else:
            pdf_list = get_all_pdf_from_dir(PATH_RAW_PDF)

        # create a folder to save the processed text
        if not os.path.exists(PATH_SAVE_TEXT):
            os.makedirs(PATH_SAVE_TEXT)

        # ------------------- CONVERT PDF TO TEXT -------------------
        print('Converting pdfs to text...', flush=True)
        # process the pdfs in parallel
        texts = process_pdfs(PATH_SAVE_TEXT, pdf_list)
    
    else:
        # get the text from the txt files
        texts, text_file_names = get_text_files_in_dir(PATH_SAVE_TEXT)

    # ------------------- FIX LINE BREAKS -------------------
    print('Fixing line breaks...', flush=True)
    texts = fix_line_breaks(texts, PATH_LOG_LINE_BREAKS)

    # ------------------- REMOVE CITATIONS -------------------
    print('Removing citations...', flush=True)
    texts = remove_citations(texts, PATH_LOG_CITATIONS)

    # ------------------- EXPAND ABBREVIATIONS -------------------
    print('Expanding abbreviations...', flush=True)
    texts, abbreviations = expand_abbreviations(texts, PATH_LOG_ABBREVIATIONS)

    # ------------------- COREFERENCE RESOLUTION -------------------
 
    if use_coref_resolution:
        # record start time
        start_time = time.time()
        print('Resolving coreferences...', flush=True)
        texts = coreference_resolution(texts, PATH_LOG_COREFERENCE)
        # record end time
        end_time = time.time()
        print('Time for coreference resolution: ', end_time - start_time, ' seconds', flush=True)
    # ------------------- SAVE TEXT -------------------
    print('Saving text...', flush=True)
    for i, text in enumerate(texts):
        if convert_pdf_to_text:
            with open(os.path.join(PATH_SAVE_TEXT, pdf_list[i].name[:-4] + ".txt"), "w", encoding="utf-8") as f:
                f.write(text)
        else:
            with open(os.path.join(PATH_SAVE_TEXT, text_file_names[i]), "w", encoding="utf-8") as f:
                f.write(text)
    # record overall end time
    end_time_overall = time.time()
    print('Overall time: ', end_time_overall - start_time_overall, ' seconds')
    print('Done!')


if __name__ == "__main__":
    main()