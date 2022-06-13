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
for mode in tqdm(["dev", "test-gold", "train"]): # ["dev", "test-gold", "train"]

    path = "/content/drive/MyDrive/temporary/masterthesis_drive/SpanEmoData/E-c/"
    filename = path + f"GoEmotions-{mode}.txt"
    df = pd.read_csv(filename, sep='\t')
    df_de = copy.deepcopy(df)
    label_order = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
    # anger, disgust, fear, joy, sadness, surprise

    # anger    anticipation     disgust     fear        joy      love         optimism    pessimism                     sadness     surprise    trust"             --> SpanEmo
    # anger                     disgust     fear        joy                                                             sadness     surprise                       --> GoEmotions Ekmann 6
    # anger    (excitement)     disgust     fear        joy      love         optimism    (disapproval/disappointment)  sadness     surprise    (approval/caring)  --> GoEmotions 28
    # anger[8] anticipation[1]  disgust[7]  fear[5]     joy[2]                                                          sadness[6]  surprise[4] trust[3]           --> Labelling

    # plutchnik28 = set(["admiration", "adoration", "aesthetic appreciation", " amusement", "anger", "anxiety", "awe", "awkwardness", "boredom", " calmness", "confusion", "craving",
    #                     "disgust", "empathic pain", " entrancement", "excitement", "fear", "horror", "interest", "joy", "nostalgia", " relief", "romance", "sadness",
    #                     "satisfaction", "sexual desire", "surprise"])
    # SpanEmpo = set(["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"])
    # GoEmotions28 = set(["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    #                  "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    #                  "sadness", "surprise", "neutral"])
    # print("len(GoEmotions28) ", len(GoEmotions28), "len(plutchnik) ", len(plutchnik28), "len(SpanEmpo) ", len(SpanEmpo))
    # print("Schnittmenge (GoEmotions28 n pluchtnik28): ", len(GoEmotions28.intersection(plutchnik28)), "\n", GoEmotions28.intersection(plutchnik28))
    # print("Schnittmenge (GoEmotions n SpanEmpo): ", len(SpanEmpo.intersection(GoEmotions28)), "\n", SpanEmpo.intersection(GoEmotions28))
    # print("Difference (SpanEmo - (GoEmotions n SpanEmo): ", len(SpanEmpo - SpanEmpo.intersection(GoEmotions28)), "\n", SpanEmpo - SpanEmpo.intersection(GoEmotions28))

    df_new = df[[label_order[0]]]
    for label in label_order[1:]:
        if label in ['trust', 'anticipation', 'pessimism']:
            df_new[label] = [0 for _ in range(df.shape[0])]
        else:
            df_new[label] = df[label]

    ###### Translating using the google api test credits ######

    # examples_text_a_de = process_map(translate_text, df["Tweet"].tolist(), max_workers=os.cpu_count())
    #
    # for i in range(len(examples_text_a_de)):
    #     df_de.iloc[i, 1] = examples_text_a_de[i]

    df_de.to_csv("new-" + filename, sep="\t", index=False)

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
