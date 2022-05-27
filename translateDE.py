import os
import pandas as pd
import pickle
import copy

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/felix/masterthesis/local-storm-351416-b1a2b33b3abc.json"

import six
from google.cloud import translate_v2 as translate

def translate_text(text, target="de"):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    return result["translatedText"]

# Translation

# text_a contains information, text_b always None
for mode in tqdm(["train", "test-gold"]): # ["dev", "test-gold", "train"]

    path = "/home/felix/masterthesis/SpanEmo/data/SemEval2018-Task1-all-data/English/E-c/"
    filename = path + f"2018-E-c-En-{mode}.txt"
    df = pd.read_csv(filename, sep='\t')
    df_de = copy.deepcopy(df)


    ###### Translating using the google api test credits ######

    examples_text_a_de = process_map(translate_text, df["Tweet"].tolist(), max_workers=os.cpu_count())

    for i in range(len(examples_text_a_de)):
        df_de.iloc[i, 1] = examples_text_a_de[i]

    df_de.to_csv(path + f"2018-E-c-En-{mode}-DE.txt", sep="\t")

    ###### Load translated obj #####

    # print("\n", mode, "\n\n")
    # with open(f"/home/felix/PycharmProjects/GoEmotions-pytorch/data/{taxonomy}/{mode}DE.obj", "rb") as f:
    #     examples_de = pickle.load(f)

    ##### Save in tsv format

    # with open(f"/home/felix/PycharmProjects/GoEmotions-pytorch/data/{taxonomy}/{mode}.tsv", "r", encoding="utf-8") as f:
    #     examples_txt =  f.readlines()
    #
    # with open(f"/home/felix/PycharmProjects/GoEmotions-pytorch/data/{taxonomy}/{mode}DE.tsv", "r", encoding="utf-8") as f:
    #     examples_txt_de = f.readlines()
    #
    # print(all([True if e.split("\t")[1:]==e_de.split("\t")[1:] else False for e, e_de in zip(examples_txt, examples_txt_de)]))
    # print("\n")

    # for i, e_t in tqdm(enumerate(examples_txt)):
    #     e_t = e_t.split("\t")
    #     if examples_de[i].label == [int(n) for n in e_t[1].split(",")] and \
    #             i == int(examples_de[i].guid.split("-")[-1]):
    #         e_t[0] = examples_de[i].text_a
    #     else:
    #         print("debug")
    #     with open(f'/home/felix/PycharmProjects/GoEmotions-pytorch/data/original/{mode}DE.tsv', 'a') as the_file:
    #         the_file.write("\t".join(e_t))


    ###### Correct for weird tokens, manual check and correction if necessary ######

    # import re
    # pattern = r"\[[a-zA-ZäüöÄÜÖ]+\]"
    # print(set(re.findall(pattern, "".join([str(e.text_a) for e in examples]))))
    # print(set(re.findall(pattern, "".join([str(e.text_a) for e in examples_de]))))
    #
    # placeholders = set(re.findall(pattern, "".join([str(e.text_a) for e in examples_de])))
    #
    # for placeholder in placeholders:
    #     elements = {i:e for i, e  in enumerate(examples_de) if placeholder in e.text_a}
    #     print(placeholder, " has ", len(elements), "cases")
    #     for i in elements.keys():
    #         print(examples[i].text_a)
    #         print(examples_de[i].text_a)
    #         text = "Warum ist Indien nicht verboten? Eines der Länder mit den meisten [RELIGION] auf dem Planeten."
    #         examples_de[i].text_a = text
    #
    # with open(f"/home/felix/PycharmProjects/GoEmotions-pytorch/data/{taxonomy}/{mode}DE.obj", "wb") as f:
    #     pickle.dump(examples_de, f)

# from transformers import AutoModel, AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
# model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
#
