import re
import emoji
import tqdm
import glob
import nltk
import string
import numpy as np
import pandas as pd
from cleantext import clean
from termcolor import cprint
from collections import Counter
from keras.backend import dtype
from nltk.corpus import stopwords
from googletrans import Translator
from Analysis import TP_COMPARTIVE
from nltk.stem import WordNetLemmatizer
from feat_ext import feature_extraction
from skimage.feature import greycomatrix
from skimage.color.rgb_colors import greenyellow

nltk.download('wordnet')
nltk.download('stopwords')

def hate_detection(db, execute):

    cprint("CSV dataset is Reading", on_color='on_grey')
    file = pd.read_csv("Training/MultiLanguageTrainDataset.csv")
    J = file["text"]
    lab = file["label"]

    #Preprocessing
    def remove_less_frequency(text):

        # words = re.findall(r'\b\w+\b')
        word_counts = Counter(text)

        threshold = 2
        filtered_words = [word for word in word_counts if word_counts[word] >= threshold]

        # filtered_text = ' '.join(filtered_words)
        print("Filtered Text:", filtered_words)
        return word_counts



    from tqdm import tqdm  # Make sure to import tqdm if you want to use it for progress bars


    def remove_punctuation(text):
        rem_pun = []
        punctuation = ":?;!'"
        for i in tqdm(text):
            cleaned_chars = [char for char in i if char not in punctuation]
            cleaned_string = ''.join(cleaned_chars)
            rem_pun.append(cleaned_string)
        return rem_pun


    def remove_using_emoji(text):
        remo_emoji = []
        for i in tqdm(text):
            rem_emo = emoji.replace_emoji(i, '')
            remo_emoji.append(rem_emo)
        return remo_emoji


    def Text_cleaning(text):
        text_clea = []
        for i in tqdm(text):
            # Lowercasing
            text = i.lower()

            # Remove URLs
            text = re.sub(r'http\S+', '', text)

            # Remove emails
            text = re.sub(r'\S*@\S*\s?', '', text)

            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            text = ' '.join([word for word in text.split() if word not in stop_words])

            # Remove numbers
            text = re.sub(r'\d+', '', text)

            # Remove extra whitespace
            text = re.sub(' +', ' ', text)

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
            text_clea.append(text)
        return text_clea


    def transliterate(text):
        english_translate = []
        for i in tqdm(text):
            try:
                translation = translator.translate(i.strip()).text
            except:
                translation = i
            english_translate.append(translation)
        return english_translate

    cprint('preprocessing', on_color='on_grey')
    translator = Translator()

    cprint("remove_less_frequency", on_color="on_grey")
    A1 = remove_less_frequency(J)

    cprint("remove_punctuation", on_color="on_grey")
    A2 = remove_punctuation(A1)

    cprint("remove_using_emoji", on_color="on_grey")
    A3 = remove_using_emoji(A2)

    cprint("Text_cleaning", on_color="on_grey")
    A4 = Text_cleaning(A3)
    A44 = A4[:1000]
    cprint("transliterate", on_color='on_grey')
    A5 = transliterate(A44)

    cprint("feature and label convert to array format", on_color='on_grey')
    feature_extr = feature_extraction(A5)
    feat = np.array(feature_extr)
    labb = np.array(lab)
    labb = labb[:1000]
    feature = np.save('feature.npy', feat)
    label = np.save('label.npy', labb)
    return feature, label


    elif db == 2:
        parquet_file_path = 'validation-00000-of-00001.parquet'
        df = pd.read_parquet(parquet_file_path)
        J = df["tweet"]
        lab = df["label"]

        def remove_less_frequency(text):

            # words = re.findall(r'\b\w+\b')
            word_counts = Counter(text)

            threshold = 2
            filtered_words = [word for word in word_counts if word_counts[word] >= threshold]

            # filtered_text = ' '.join(filtered_words)
            print("Filtered Text:", filtered_words)
            return word_counts

            # def remove_punctuation(my_string):
            #     # Sample string
            #     # my_string = "Hello! How are you? I'm doing well, thanks."

            #     # Remove punctuation
            #     new_string = ""
            #     for char in my_string:
            #         if char not in string.punctuation:
            #             new_string += char
            #     # Output
            #     print(new_string)
            #     return new_string

        from tqdm import tqdm  # Make sure to import tqdm if you want to use it for progress bars

        def remove_punctuation(text):
            rem_pun = []
            punctuation = ":?;!'"
            for i in tqdm(text):
                cleaned_chars = [char for char in i if char not in punctuation]
                cleaned_string = ''.join(cleaned_chars)
                rem_pun.append(cleaned_string)
            return rem_pun

        def remove_using_emoji(text):
            remo_emoji = []
            for i in tqdm(text):
                rem_emo = emoji.replace_emoji(i, '')
                remo_emoji.append(rem_emo)
            return remo_emoji

        def Text_cleaning(text):
            text_clea = []
            for i in tqdm(text):
                # Lowercasing
                text = i.lower()

                # Remove URLs
                text = re.sub(r'http\S+', '', text)

                # Remove emails
                text = re.sub(r'\S*@\S*\s?', '', text)

                # Remove punctuation
                text = text.translate(str.maketrans('', '', string.punctuation))

                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                text = ' '.join([word for word in text.split() if word not in stop_words])

                # Remove numbers
                text = re.sub(r'\d+', '', text)

                # Remove extra whitespace
                text = re.sub(' +', ' ', text)

                # Lemmatization
                lemmatizer = WordNetLemmatizer()
                text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
                text_clea.append(text)
            return text_clea

        def transliterate(text):
            english_translate = []
            for i in tqdm(text):
                try:
                    translation = translator.translate(i.strip()).text
                except:
                    translation = i
                english_translate.append(translation)
            return english_translate

        cprint('preprocessing', on_color='on_grey')
        translator = Translator()

        cprint("remove_less_frequency", on_color="on_grey")
        A1 = remove_less_frequency(J)

        cprint("remove_punctuation", on_color="on_grey")
        A2 = remove_punctuation(A1)

        cprint("remove_using_emoji", on_color="on_grey")
        A3 = remove_using_emoji(A2)

        cprint("Text_cleaning", on_color="on_grey")
        A4 = Text_cleaning(A3)
        A44 = A4[:1000]
        cprint("transliterate", on_color='on_grey')
        A5 = transliterate(A44)

        # for i in A5[:1]:
        #     # feature extraction
        cprint("feature and label convert to array format", on_color='on_grey')
        feature_extr = feature_extraction(A5)
        feat = np.array(feature_extr)
        labb = np.array(lab)
        labb = labb[:1000]
        feature = np.save('feature2.npy', feat)
        label = np.save('lab2.npy', labb)
        return feature, label


    # else:
    #     if db == 1:
    #         FEAT = np.load('feature.npy', allow_pickle=True)
    #         LAB = np.load('label.npy', allow_pickle=True)
    #         return FEAT, LAB.astype(int)
    #
    #     else:
    #         FEATT = np.load('feature2.npy')
    #         LAABB = np.load('lab2.npy')
    #         # FEAT1 = FEAT1[1:, :]
    #         return FEATT, LAABB


d = [2]
for i in d:
    FEAT, LAB = mainfile(i, True)
    TP_COMPARTIVE(FEAT, LAB, i)