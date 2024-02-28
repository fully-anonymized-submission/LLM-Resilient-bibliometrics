import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import fitz

import os
import pandas as pd
from typing import Union

from pathlib import Path
import random

import nltk

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

    if os.path.exists(PATH_METADATA):
        df = pd.read_pickle(PATH_METADATA)
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

def main():
    """Convert pdfs to text and process the text"""
    ###################################   SETTINGS  ###################################################
    inverse = False # put to true if you want to remove categories instead of keeping them
    subset = None # if you want to process only a subset of the pdfs, put the number here
    filter_categories = True # if you want to filter based on the categories, put to True
    filter_by_version = False # whether to keep only the newest version of each paper
    categories = ['cs.AI', 'cs.CL', 'cs.LG'] # categories to keep (if inverse is False) or to remove (if inverse is True)

    # please run from the root directory of the project
    PATH_ROOT = os.getcwd()
    print('Current working directory: ', PATH_ROOT)

    ####################################### FILL IN THE PATHS ########################################
    PATH_METADATA = PATH_ROOT + '' # path to the metadata, should be a .pickle file
    PATH_RAW_PDF = PATH_ROOT + '' # path to the directory with the raw pdfs
    PATH_SAVE_TEXT = PATH_ROOT + '' # path to the directory to save the processed text
    ##################################################################################################

    if not os.path.exists(PATH_SAVE_TEXT):
        print('Creating folder to save texts... ')
        os.makedirs(PATH_SAVE_TEXT)
    # ------------------- KEEP NEWEST PAPER VERSION -------------------
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


    # create a folder to save the processed text
    if not os.path.exists(PATH_SAVE_TEXT):
        os.makedirs(PATH_SAVE_TEXT)

    # ------------------- CONVERT PDF TO TEXT -------------------
    print('Converting pdfs to text...', flush=True)
    # process the pdfs in parallel
    texts = process_pdfs(PATH_SAVE_TEXT, pdf_list)
    
if __name__ == "__main__":
    main()