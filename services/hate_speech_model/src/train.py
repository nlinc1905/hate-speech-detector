import pandas as pd
import os
import re
import requests
from io import BytesIO
from zipfile import ZipFile

from model import HateSpeechClassifier


# url obtained from https://github.com/stanfordnlp/GloVe
glove_url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip"
# url obtained from https://fasttext.cc/docs/en/english-vectors.html
fasttext_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"


def download_and_extract_word_vectors(url: str, model_path: str):
    # if the vectors already exist, do nothing
    model_name = ".".join(url.split("/")[-1].split(".")[:-1])
    # look for files ending in .txt for glove or .vec for fasttext
    if model_name.split(".")[-1] != "vec":
        model_name += ".txt"
    regex = re.compile(f"(.*{model_name.split('.')[-1]}$)")  # re.compile("(.*vec$)|(.*txt$)")
    exists = False
    for _, _, files in os.walk(model_path):
        for file in files:
            if regex.match(file):
                print(f"Word vectors found: {file}, skipping.")
                exists = True

    if not exists:
        # get the resource from the URL
        resp = requests.get(url, stream=True)
        zf = ZipFile(BytesIO(resp.content))

        # extract to given path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        zf.extractall(path=model_path)


def trained_model_exists(save_path: str):
    """Returns True if a trained model already exists."""
    model_regex = re.compile("(.*pt$)")
    tokenizer_regex = re.compile("(.*pkl$)")
    model_condition, tokenizer_condition = False, False
    for _, _, files in os.walk(save_path):
        for file in files:
            if model_regex.match(file):
                model_condition = True
            if tokenizer_regex.match(file):
                tokenizer_condition = True

    if model_condition and tokenizer_condition:
        return True
    else:
        return False


if __name__ == "__main__":
    # get word vectors
    download_and_extract_word_vectors(url=glove_url, model_path="word_vectors/")
    download_and_extract_word_vectors(url=fasttext_url, model_path="word_vectors/")

    # train classifier
    if not trained_model_exists(save_path='models/'):
        cls = HateSpeechClassifier()
        cls.train(train_data_path='data/jigsaw_unintended_bias/train.csv')
        cls.save_classifier(save_path='models/')

    # uncomment lines below to debug
    # cls = HateSpeechClassifier()
    # cls.load_classifier(load_path='models/')
    # test_data = pd.read_csv('data/jigsaw_unintended_bias/test.csv').to_dict(orient="records")
    # test_preds = cls.predict(test_data=test_data)
    # breakpoint()
