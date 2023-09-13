def check_backtrack(x):  # line 1
    if not (x == 'a'  # line 2
        or x == 'c'):  # line 3
        pass  # line 4


import re
import sys

en_lang_symbols = r'[^\w!@#$%\^-_+=|\}{][\"\';:?\/><.,&)(*\s`\u2019]'
en_words_basic = []
en_words = []


class Dummy:
    non_en_words_limit = 3

    @staticmethod
    def fun(text):
        words = tuple(w[0].lower() for w in re.finditer(r'[a-zA-Z]+', text))
        non_en_pass = []
        for i, word in enumerate(words):
            non_en = []
            if not (word in en_words_basic
                    or (word.endswith('s') and word[:-1] in en_words_basic)
                    or (word.endswith('ed') and word[:-2] in en_words_basic)
                    or (word.endswith('ing') and word[:-3] in en_words_basic)
                    or word in en_words
                    or (word.endswith('s') and word[:-1] in en_words)
                    or (word.endswith('ed') and word[:-2] in en_words)
                    or (word.endswith('ing') and word[:-3] in en_words)
                    ):

                non_en.append(word)
                non_en_pass.append(word)
                for j in range(1, Dummy.non_en_words_limit):
                    if i + j >= len(words):
                        break
                    word = words[i + j]

                    if (word in en_words_basic
                        or (word.endswith('s') and word[:-1] in en_words_basic)
                        or (word.endswith('ed') and word[:-2] in en_words_basic)
                        or (word.endswith('ing') and word[:-3] in en_words_basic)
                        or word in en_words
                        or (word.endswith('s') and word[:-1] in en_words)
                        or (word.endswith('ed') and word[:-2] in en_words)
                        or (word.endswith('ing') and word[:-3] in en_words)
                    ):
                        break
                    else:
                        non_en.append(word)
                        non_en_pass.append(word)


def offset_overflow(stream=sys.stdout):
    a = 1
    b = 2
    c = 3
    a1 = 1 if a > 1 else 2
    a2 = 1 if a > 1 else 2
    a3 = 1 if a > 1 else 2
    a4 = 1 if a > 1 else 2
    a5 = 1 if a > 1 else 2
    a6 = 1 if a > 1 else 2
    a7 = 1 if a > 1 else 2
    a8 = 1 if a > 1 else 2
    a9 = 1 if a > 1 else 2
    a10 = 1 if a > 1 else 2
    a11 = 1 if a > 1 else 2
    a12 = 1 if a > 1 else 2
    a13 = 1 if a > 1 else 2

    for i in range(1):
        if a > 0:
            stream.write("111\n")
            # a = 1
        else:
            stream.write("222\n")
    return b


def long_lines():
    a = 1
    b = 1 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23
    c = 1 if b > 1 else 2 if b > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23
    d = 1 if c > 1 else 2 if c > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23 if a > 1 else 2 if a > 0 else 3 if a > 4 else 23
    e = d + 1
    return e
