#######################################################################
# This file implements the required function for cleaning a text
# الله المستعان
#######################################################################

import re

def remove_unwanted_chars(text):
    unwanted = r"[^a-zA-Z0-9?.,!\s★']"    
    return re.sub(unwanted, "", text)



def clean_punctuation(text):
    return re.sub(r"([.?!,])\1+", r"\1", text)


def remove_links(text):
    links = r"http\S|www\S+"
    mentions = r"@\W+"
    text = re.sub(links, "", text)
    text = re.sub(mentions, "", text)
    return text


def to_lower(text):
    return text.lower()

def std_spaces(text):
    return re.sub(r"\s+", " ", text)


def parse_html_tags(text):
    from bs4 import BeautifulSoup
    return (BeautifulSoup(text, "html.parser").get_text(separator=" "))


def clean_text(text):
    cleaned_text = remove_links(text)
    cleaned_text = parse_html_tags(cleaned_text)
    cleaned_text = clean_punctuation(cleaned_text)
    cleaned_text = remove_unwanted_chars(cleaned_text)
    cleaned_text = to_lower(cleaned_text)
    cleaned_text = std_spaces(cleaned_text)
    return cleaned_text



def get_unique_chars(data: list):
    sentences = "".join(data)
    unique_chars = set([c.lower() for c in sentences])
    return unique_chars