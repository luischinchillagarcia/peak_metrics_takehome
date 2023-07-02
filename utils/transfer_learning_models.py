from __future__ import annotations
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


@dataclass
class SentimentTask:
    model = pipeline(
        "sentiment-analysis", 
        model=BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3), # type: ignore
        tokenizer=BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis"),
    ) 


@dataclass
class EmotionTask:
    model = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)



@dataclass
class EmotionTaskLarge:
    model = pipeline("text-classification", model='SamLowe/roberta-base-go_emotions', return_all_scores=True)


@dataclass
class NERModel:
    # Contains 4 entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC).
    model = pipeline(
        "ner", 
        model=AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner"), 
        tokenizer=AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner"), 
        grouped_entities=True,
    )


# Due to time, computation constraints, it's best to leave translation for a different time
# Use this import: from transformers import AutoModelForSeq2SeqLM
# @dataclass
# class TranslationModel:
#     tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100")
#     model = AutoModelForSeq2SeqLM.from_pretrained("alirezamsh/small100")


# def translate_to_en(text):
#     TranslationModel.tokenizer.tgt_lang = "en" # type: ignore
#     encoded_hi = TranslationModel.tokenizer(text, return_tensors="pt")
#     generated_tokens = TranslationModel.model.generate(**encoded_hi)
#     translated_text = TranslationModel.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#     return translated_text
