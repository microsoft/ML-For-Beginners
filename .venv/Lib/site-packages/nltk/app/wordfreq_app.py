# Natural Language Toolkit: Wordfreq Application
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Sumukh Ghodke <sghodke@csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from matplotlib import pylab

from nltk.corpus import gutenberg
from nltk.text import Text


def plot_word_freq_dist(text):
    fd = text.vocab()

    samples = [item for item, _ in fd.most_common(50)]
    values = [fd[sample] for sample in samples]
    values = [sum(values[: i + 1]) * 100.0 / fd.N() for i in range(len(values))]
    pylab.title(text.name)
    pylab.xlabel("Samples")
    pylab.ylabel("Cumulative Percentage")
    pylab.plot(values)
    pylab.xticks(range(len(samples)), [str(s) for s in samples], rotation=90)
    pylab.show()


def app():
    t1 = Text(gutenberg.words("melville-moby_dick.txt"))
    plot_word_freq_dist(t1)


if __name__ == "__main__":
    app()

__all__ = ["app"]
