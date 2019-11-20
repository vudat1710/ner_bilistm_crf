import string

NUM = "<NUM>"
PUNCT = "<PUNCT>"

def get_processing_word(lowercase=False):
    def f(word):
        if lowercase:
            word = word.lower()
        if word[0].isdigit():
            word = NUM
        if word in string.punctuation and word != "_":
            word = PUNCT
        return word

    return f