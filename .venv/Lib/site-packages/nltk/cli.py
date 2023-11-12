# Natural Language Toolkit: NLTK Command-Line Interface
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


import click
from tqdm import tqdm

from nltk import word_tokenize
from nltk.util import parallelize_preprocess

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli():
    pass


@cli.command("tokenize")
@click.option(
    "--language",
    "-l",
    default="en",
    help="The language for the Punkt sentence tokenization.",
)
@click.option(
    "--preserve-line",
    "-l",
    default=True,
    is_flag=True,
    help="An option to keep the preserve the sentence and not sentence tokenize it.",
)
@click.option("--processes", "-j", default=1, help="No. of processes.")
@click.option("--encoding", "-e", default="utf8", help="Specify encoding of file.")
@click.option(
    "--delimiter", "-d", default=" ", help="Specify delimiter to join the tokens."
)
def tokenize_file(language, preserve_line, processes, encoding, delimiter):
    """This command tokenizes text stream using nltk.word_tokenize"""
    with click.get_text_stream("stdin", encoding=encoding) as fin:
        with click.get_text_stream("stdout", encoding=encoding) as fout:
            # If it's single process, joblib parallelization is slower,
            # so just process line by line normally.
            if processes == 1:
                for line in tqdm(fin.readlines()):
                    print(delimiter.join(word_tokenize(line)), end="\n", file=fout)
            else:
                for outline in parallelize_preprocess(
                    word_tokenize, fin.readlines(), processes, progress_bar=True
                ):
                    print(delimiter.join(outline), end="\n", file=fout)
