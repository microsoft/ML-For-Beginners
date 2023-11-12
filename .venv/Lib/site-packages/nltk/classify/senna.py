# Natural Language Toolkit: Senna Interface
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Rami Al-Rfou' <ralrfou@cs.stonybrook.edu>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
A general interface to the SENNA pipeline that supports any of the
operations specified in SUPPORTED_OPERATIONS.

Applying multiple operations at once has the speed advantage. For example,
Senna will automatically determine POS tags if you are extracting named
entities. Applying both of the operations will cost only the time of
extracting the named entities.

The SENNA pipeline has a fixed maximum size of the sentences that it can read.
By default it is 1024 token/sentence. If you have larger sentences, changing
the MAX_SENTENCE_SIZE value in SENNA_main.c should be considered and your
system specific binary should be rebuilt. Otherwise this could introduce
misalignment errors.

The input is:

- path to the directory that contains SENNA executables. If the path is incorrect,
  Senna will automatically search for executable file specified in SENNA environment variable
- List of the operations needed to be performed.
- (optionally) the encoding of the input data (default:utf-8)

Note: Unit tests for this module can be found in test/unit/test_senna.py

>>> from nltk.classify import Senna
>>> pipeline = Senna('/usr/share/senna-v3.0', ['pos', 'chk', 'ner'])  # doctest: +SKIP
>>> sent = 'Dusseldorf is an international business center'.split()
>>> [(token['word'], token['chk'], token['ner'], token['pos']) for token in pipeline.tag(sent)]  # doctest: +SKIP
[('Dusseldorf', 'B-NP', 'B-LOC', 'NNP'), ('is', 'B-VP', 'O', 'VBZ'), ('an', 'B-NP', 'O', 'DT'),
('international', 'I-NP', 'O', 'JJ'), ('business', 'I-NP', 'O', 'NN'), ('center', 'I-NP', 'O', 'NN')]
"""

from os import environ, path, sep
from platform import architecture, system
from subprocess import PIPE, Popen

from nltk.tag.api import TaggerI


class Senna(TaggerI):

    SUPPORTED_OPERATIONS = ["pos", "chk", "ner"]

    def __init__(self, senna_path, operations, encoding="utf-8"):
        self._encoding = encoding
        self._path = path.normpath(senna_path) + sep

        # Verifies the existence of the executable on the self._path first
        # senna_binary_file_1 = self.executable(self._path)
        exe_file_1 = self.executable(self._path)
        if not path.isfile(exe_file_1):
            # Check for the system environment
            if "SENNA" in environ:
                # self._path = path.join(environ['SENNA'],'')
                self._path = path.normpath(environ["SENNA"]) + sep
                exe_file_2 = self.executable(self._path)
                if not path.isfile(exe_file_2):
                    raise LookupError(
                        "Senna executable expected at %s or %s but not found"
                        % (exe_file_1, exe_file_2)
                    )

        self.operations = operations

    def executable(self, base_path):
        """
        The function that determines the system specific binary that should be
        used in the pipeline. In case, the system is not known the default senna binary will
        be used.
        """
        os_name = system()
        if os_name == "Linux":
            bits = architecture()[0]
            if bits == "64bit":
                return path.join(base_path, "senna-linux64")
            return path.join(base_path, "senna-linux32")
        if os_name == "Windows":
            return path.join(base_path, "senna-win32.exe")
        if os_name == "Darwin":
            return path.join(base_path, "senna-osx")
        return path.join(base_path, "senna")

    def _map(self):
        """
        A method that calculates the order of the columns that SENNA pipeline
        will output the tags into. This depends on the operations being ordered.
        """
        _map = {}
        i = 1
        for operation in Senna.SUPPORTED_OPERATIONS:
            if operation in self.operations:
                _map[operation] = i
                i += 1
        return _map

    def tag(self, tokens):
        """
        Applies the specified operation(s) on a list of tokens.
        """
        return self.tag_sents([tokens])[0]

    def tag_sents(self, sentences):
        """
        Applies the tag method over a list of sentences. This method will return a
        list of dictionaries. Every dictionary will contain a word with its
        calculated annotations/tags.
        """
        encoding = self._encoding

        if not path.isfile(self.executable(self._path)):
            raise LookupError(
                "Senna executable expected at %s but not found"
                % self.executable(self._path)
            )

        # Build the senna command to run the tagger
        _senna_cmd = [
            self.executable(self._path),
            "-path",
            self._path,
            "-usrtokens",
            "-iobtags",
        ]
        _senna_cmd.extend(["-" + op for op in self.operations])

        # Serialize the actual sentences to a temporary string
        _input = "\n".join(" ".join(x) for x in sentences) + "\n"
        if isinstance(_input, str) and encoding:
            _input = _input.encode(encoding)

        # Run the tagger and get the output
        p = Popen(_senna_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        (stdout, stderr) = p.communicate(input=_input)
        senna_output = stdout

        # Check the return code.
        if p.returncode != 0:
            raise RuntimeError("Senna command failed! Details: %s" % stderr)

        if encoding:
            senna_output = stdout.decode(encoding)

        # Output the tagged sentences
        map_ = self._map()
        tagged_sentences = [[]]
        sentence_index = 0
        token_index = 0
        for tagged_word in senna_output.strip().split("\n"):
            if not tagged_word:
                tagged_sentences.append([])
                sentence_index += 1
                token_index = 0
                continue
            tags = tagged_word.split("\t")
            result = {}
            for tag in map_:
                result[tag] = tags[map_[tag]].strip()
            try:
                result["word"] = sentences[sentence_index][token_index]
            except IndexError as e:
                raise IndexError(
                    "Misalignment error occurred at sentence number %d. Possible reason"
                    " is that the sentence size exceeded the maximum size. Check the "
                    "documentation of Senna class for more information."
                    % sentence_index
                ) from e
            tagged_sentences[-1].append(result)
            token_index += 1
        return tagged_sentences
