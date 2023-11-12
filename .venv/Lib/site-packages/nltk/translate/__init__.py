# Natural Language Toolkit: Machine Translation
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>, Tah Wei Hoon <hoon.tw@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Experimental features for machine translation.
These interfaces are prone to change.

isort:skip_file
"""

from nltk.translate.api import AlignedSent, Alignment, PhraseTable
from nltk.translate.ibm_model import IBMModel
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2
from nltk.translate.ibm3 import IBMModel3
from nltk.translate.ibm4 import IBMModel4
from nltk.translate.ibm5 import IBMModel5
from nltk.translate.bleu_score import sentence_bleu as bleu
from nltk.translate.ribes_score import sentence_ribes as ribes
from nltk.translate.meteor_score import meteor_score as meteor
from nltk.translate.metrics import alignment_error_rate
from nltk.translate.stack_decoder import StackDecoder
from nltk.translate.nist_score import sentence_nist as nist
from nltk.translate.chrf_score import sentence_chrf as chrf
from nltk.translate.gale_church import trace
from nltk.translate.gdfa import grow_diag_final_and
from nltk.translate.gleu_score import sentence_gleu as gleu
from nltk.translate.phrase_based import extract
