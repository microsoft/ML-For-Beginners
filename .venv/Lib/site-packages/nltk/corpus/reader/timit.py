# Natural Language Toolkit: TIMIT Corpus Reader
#
# Copyright (C) 2001-2007 NLTK Project
# Author: Haejoong Lee <haejoong@ldc.upenn.edu>
#         Steven Bird <stevenbird1@gmail.com>
#         Jacob Perkins <japerk@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

# [xx] this docstring is out-of-date:
"""
Read tokens, phonemes and audio data from the NLTK TIMIT Corpus.

This corpus contains selected portion of the TIMIT corpus.

 - 16 speakers from 8 dialect regions
 - 1 male and 1 female from each dialect region
 - total 130 sentences (10 sentences per speaker.  Note that some
   sentences are shared among other speakers, especially sa1 and sa2
   are spoken by all speakers.)
 - total 160 recording of sentences (10 recordings per speaker)
 - audio format: NIST Sphere, single channel, 16kHz sampling,
   16 bit sample, PCM encoding


Module contents
===============

The timit corpus reader provides 4 functions and 4 data items.

 - utterances

   List of utterances in the corpus.  There are total 160 utterances,
   each of which corresponds to a unique utterance of a speaker.
   Here's an example of an utterance identifier in the list::

       dr1-fvmh0/sx206
         - _----  _---
         | |  |   | |
         | |  |   | |
         | |  |   | `--- sentence number
         | |  |   `----- sentence type (a:all, i:shared, x:exclusive)
         | |  `--------- speaker ID
         | `------------ sex (m:male, f:female)
         `-------------- dialect region (1..8)

 - speakers

   List of speaker IDs.  An example of speaker ID::

       dr1-fvmh0

   Note that if you split an item ID with colon and take the first element of
   the result, you will get a speaker ID.

       >>> itemid = 'dr1-fvmh0/sx206'
       >>> spkrid , sentid = itemid.split('/')
       >>> spkrid
       'dr1-fvmh0'

   The second element of the result is a sentence ID.

 - dictionary()

   Phonetic dictionary of words contained in this corpus.  This is a Python
   dictionary from words to phoneme lists.

 - spkrinfo()

   Speaker information table.  It's a Python dictionary from speaker IDs to
   records of 10 fields.  Speaker IDs the same as the ones in timie.speakers.
   Each record is a dictionary from field names to values, and the fields are
   as follows::

     id         speaker ID as defined in the original TIMIT speaker info table
     sex        speaker gender (M:male, F:female)
     dr         speaker dialect region (1:new england, 2:northern,
                3:north midland, 4:south midland, 5:southern, 6:new york city,
                7:western, 8:army brat (moved around))
     use        corpus type (TRN:training, TST:test)
                in this sample corpus only TRN is available
     recdate    recording date
     birthdate  speaker birth date
     ht         speaker height
     race       speaker race (WHT:white, BLK:black, AMR:american indian,
                SPN:spanish-american, ORN:oriental,???:unknown)
     edu        speaker education level (HS:high school, AS:associate degree,
                BS:bachelor's degree (BS or BA), MS:master's degree (MS or MA),
                PHD:doctorate degree (PhD,JD,MD), ??:unknown)
     comments   comments by the recorder

The 4 functions are as follows.

 - tokenized(sentences=items, offset=False)

   Given a list of items, returns an iterator of a list of word lists,
   each of which corresponds to an item (sentence).  If offset is set to True,
   each element of the word list is a tuple of word(string), start offset and
   end offset, where offset is represented as a number of 16kHz samples.

 - phonetic(sentences=items, offset=False)

   Given a list of items, returns an iterator of a list of phoneme lists,
   each of which corresponds to an item (sentence).  If offset is set to True,
   each element of the phoneme list is a tuple of word(string), start offset
   and end offset, where offset is represented as a number of 16kHz samples.

 - audiodata(item, start=0, end=None)

   Given an item, returns a chunk of audio samples formatted into a string.
   When the function is called, if start and end are omitted, the entire
   samples of the recording will be returned.  If only end is omitted,
   samples from the start offset to the end of the recording will be returned.

 - play(data)

   Play the given audio samples. The audio samples can be obtained from the
   timit.audiodata function.

"""
import sys
import time

from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree


class TimitCorpusReader(CorpusReader):
    """
    Reader for the TIMIT corpus (or any other corpus with the same
    file layout and use of file formats).  The corpus root directory
    should contain the following files:

      - timitdic.txt: dictionary of standard transcriptions
      - spkrinfo.txt: table of speaker information

    In addition, the root directory should contain one subdirectory
    for each speaker, containing three files for each utterance:

      - <utterance-id>.txt: text content of utterances
      - <utterance-id>.wrd: tokenized text content of utterances
      - <utterance-id>.phn: phonetic transcription of utterances
      - <utterance-id>.wav: utterance sound file
    """

    _FILE_RE = r"(\w+-\w+/\w+\.(phn|txt|wav|wrd))|" + r"timitdic\.txt|spkrinfo\.txt"
    """A regexp matching fileids that are used by this corpus reader."""
    _UTTERANCE_RE = r"\w+-\w+/\w+\.txt"

    def __init__(self, root, encoding="utf8"):
        """
        Construct a new TIMIT corpus reader in the given directory.
        :param root: The root directory for this corpus.
        """
        # Ensure that wave files don't get treated as unicode data:
        if isinstance(encoding, str):
            encoding = [(r".*\.wav", None), (".*", encoding)]

        CorpusReader.__init__(
            self, root, find_corpus_fileids(root, self._FILE_RE), encoding=encoding
        )

        self._utterances = [
            name[:-4] for name in find_corpus_fileids(root, self._UTTERANCE_RE)
        ]
        """A list of the utterance identifiers for all utterances in
        this corpus."""

        self._speakerinfo = None
        self._root = root
        self.speakers = sorted({u.split("/")[0] for u in self._utterances})

    def fileids(self, filetype=None):
        """
        Return a list of file identifiers for the files that make up
        this corpus.

        :param filetype: If specified, then ``filetype`` indicates that
            only the files that have the given type should be
            returned.  Accepted values are: ``txt``, ``wrd``, ``phn``,
            ``wav``, or ``metadata``,
        """
        if filetype is None:
            return CorpusReader.fileids(self)
        elif filetype in ("txt", "wrd", "phn", "wav"):
            return [f"{u}.{filetype}" for u in self._utterances]
        elif filetype == "metadata":
            return ["timitdic.txt", "spkrinfo.txt"]
        else:
            raise ValueError("Bad value for filetype: %r" % filetype)

    def utteranceids(
        self, dialect=None, sex=None, spkrid=None, sent_type=None, sentid=None
    ):
        """
        :return: A list of the utterance identifiers for all
            utterances in this corpus, or for the given speaker, dialect
            region, gender, sentence type, or sentence number, if
            specified.
        """
        if isinstance(dialect, str):
            dialect = [dialect]
        if isinstance(sex, str):
            sex = [sex]
        if isinstance(spkrid, str):
            spkrid = [spkrid]
        if isinstance(sent_type, str):
            sent_type = [sent_type]
        if isinstance(sentid, str):
            sentid = [sentid]

        utterances = self._utterances[:]
        if dialect is not None:
            utterances = [u for u in utterances if u[2] in dialect]
        if sex is not None:
            utterances = [u for u in utterances if u[4] in sex]
        if spkrid is not None:
            utterances = [u for u in utterances if u[:9] in spkrid]
        if sent_type is not None:
            utterances = [u for u in utterances if u[11] in sent_type]
        if sentid is not None:
            utterances = [u for u in utterances if u[10:] in spkrid]
        return utterances

    def transcription_dict(self):
        """
        :return: A dictionary giving the 'standard' transcription for
            each word.
        """
        _transcriptions = {}
        with self.open("timitdic.txt") as fp:
            for line in fp:
                if not line.strip() or line[0] == ";":
                    continue
                m = re.match(r"\s*(\S+)\s+/(.*)/\s*$", line)
                if not m:
                    raise ValueError("Bad line: %r" % line)
                _transcriptions[m.group(1)] = m.group(2).split()
        return _transcriptions

    def spkrid(self, utterance):
        return utterance.split("/")[0]

    def sentid(self, utterance):
        return utterance.split("/")[1]

    def utterance(self, spkrid, sentid):
        return f"{spkrid}/{sentid}"

    def spkrutteranceids(self, speaker):
        """
        :return: A list of all utterances associated with a given
            speaker.
        """
        return [
            utterance
            for utterance in self._utterances
            if utterance.startswith(speaker + "/")
        ]

    def spkrinfo(self, speaker):
        """
        :return: A dictionary mapping .. something.
        """
        if speaker in self._utterances:
            speaker = self.spkrid(speaker)

        if self._speakerinfo is None:
            self._speakerinfo = {}
            with self.open("spkrinfo.txt") as fp:
                for line in fp:
                    if not line.strip() or line[0] == ";":
                        continue
                    rec = line.strip().split(None, 9)
                    key = f"dr{rec[2]}-{rec[1].lower()}{rec[0].lower()}"
                    self._speakerinfo[key] = SpeakerInfo(*rec)

        return self._speakerinfo[speaker]

    def phones(self, utterances=None):
        results = []
        for fileid in self._utterance_fileids(utterances, ".phn"):
            with self.open(fileid) as fp:
                for line in fp:
                    if line.strip():
                        results.append(line.split()[-1])
        return results

    def phone_times(self, utterances=None):
        """
        offset is represented as a number of 16kHz samples!
        """
        results = []
        for fileid in self._utterance_fileids(utterances, ".phn"):
            with self.open(fileid) as fp:
                for line in fp:
                    if line.strip():
                        results.append(
                            (
                                line.split()[2],
                                int(line.split()[0]),
                                int(line.split()[1]),
                            )
                        )
        return results

    def words(self, utterances=None):
        results = []
        for fileid in self._utterance_fileids(utterances, ".wrd"):
            with self.open(fileid) as fp:
                for line in fp:
                    if line.strip():
                        results.append(line.split()[-1])
        return results

    def word_times(self, utterances=None):
        results = []
        for fileid in self._utterance_fileids(utterances, ".wrd"):
            with self.open(fileid) as fp:
                for line in fp:
                    if line.strip():
                        results.append(
                            (
                                line.split()[2],
                                int(line.split()[0]),
                                int(line.split()[1]),
                            )
                        )
        return results

    def sents(self, utterances=None):
        results = []
        for fileid in self._utterance_fileids(utterances, ".wrd"):
            with self.open(fileid) as fp:
                results.append([line.split()[-1] for line in fp if line.strip()])
        return results

    def sent_times(self, utterances=None):
        # TODO: Check this
        return [
            (
                line.split(None, 2)[-1].strip(),
                int(line.split()[0]),
                int(line.split()[1]),
            )
            for fileid in self._utterance_fileids(utterances, ".txt")
            for line in self.open(fileid)
            if line.strip()
        ]

    def phone_trees(self, utterances=None):
        if utterances is None:
            utterances = self._utterances
        if isinstance(utterances, str):
            utterances = [utterances]

        trees = []
        for utterance in utterances:
            word_times = self.word_times(utterance)
            phone_times = self.phone_times(utterance)
            sent_times = self.sent_times(utterance)

            while sent_times:
                (sent, sent_start, sent_end) = sent_times.pop(0)
                trees.append(Tree("S", []))
                while (
                    word_times and phone_times and phone_times[0][2] <= word_times[0][1]
                ):
                    trees[-1].append(phone_times.pop(0)[0])
                while word_times and word_times[0][2] <= sent_end:
                    (word, word_start, word_end) = word_times.pop(0)
                    trees[-1].append(Tree(word, []))
                    while phone_times and phone_times[0][2] <= word_end:
                        trees[-1][-1].append(phone_times.pop(0)[0])
                while phone_times and phone_times[0][2] <= sent_end:
                    trees[-1].append(phone_times.pop(0)[0])
        return trees

    # [xx] NOTE: This is currently broken -- we're assuming that the
    # fileids are WAV fileids (aka RIFF), but they're actually NIST SPHERE
    # fileids.
    def wav(self, utterance, start=0, end=None):
        # nltk.chunk conflicts with the stdlib module 'chunk'
        wave = import_from_stdlib("wave")

        w = wave.open(self.open(utterance + ".wav"), "rb")

        if end is None:
            end = w.getnframes()

        # Skip past frames before start, then read the frames we want
        w.readframes(start)
        frames = w.readframes(end - start)

        # Open a new temporary file -- the wave module requires
        # an actual file, and won't work w/ stringio. :(
        tf = tempfile.TemporaryFile()
        out = wave.open(tf, "w")

        # Write the parameters & data to the new file.
        out.setparams(w.getparams())
        out.writeframes(frames)
        out.close()

        # Read the data back from the file, and return it.  The
        # file will automatically be deleted when we return.
        tf.seek(0)
        return tf.read()

    def audiodata(self, utterance, start=0, end=None):
        assert end is None or end > start
        headersize = 44
        with self.open(utterance + ".wav") as fp:
            if end is None:
                data = fp.read()
            else:
                data = fp.read(headersize + end * 2)
        return data[headersize + start * 2 :]

    def _utterance_fileids(self, utterances, extension):
        if utterances is None:
            utterances = self._utterances
        if isinstance(utterances, str):
            utterances = [utterances]
        return [f"{u}{extension}" for u in utterances]

    def play(self, utterance, start=0, end=None):
        """
        Play the given audio sample.

        :param utterance: The utterance id of the sample to play
        """
        # Method 1: os audio dev.
        try:
            import ossaudiodev

            try:
                dsp = ossaudiodev.open("w")
                dsp.setfmt(ossaudiodev.AFMT_S16_LE)
                dsp.channels(1)
                dsp.speed(16000)
                dsp.write(self.audiodata(utterance, start, end))
                dsp.close()
            except OSError as e:
                print(
                    (
                        "can't acquire the audio device; please "
                        "activate your audio device."
                    ),
                    file=sys.stderr,
                )
                print("system error message:", str(e), file=sys.stderr)
            return
        except ImportError:
            pass

        # Method 2: pygame
        try:
            # FIXME: this won't work under python 3
            import pygame.mixer
            import StringIO

            pygame.mixer.init(16000)
            f = StringIO.StringIO(self.wav(utterance, start, end))
            pygame.mixer.Sound(f).play()
            while pygame.mixer.get_busy():
                time.sleep(0.01)
            return
        except ImportError:
            pass

        # Method 3: complain. :)
        print(
            ("you must install pygame or ossaudiodev " "for audio playback."),
            file=sys.stderr,
        )


class SpeakerInfo:
    def __init__(
        self, id, sex, dr, use, recdate, birthdate, ht, race, edu, comments=None
    ):
        self.id = id
        self.sex = sex
        self.dr = dr
        self.use = use
        self.recdate = recdate
        self.birthdate = birthdate
        self.ht = ht
        self.race = race
        self.edu = edu
        self.comments = comments

    def __repr__(self):
        attribs = "id sex dr use recdate birthdate ht race edu comments"
        args = [f"{attr}={getattr(self, attr)!r}" for attr in attribs.split()]
        return "SpeakerInfo(%s)" % (", ".join(args))


def read_timit_block(stream):
    """
    Block reader for timit tagged sentences, which are preceded by a sentence
    number that will be ignored.
    """
    line = stream.readline()
    if not line:
        return []
    n, sent = line.split(" ", 1)
    return [sent]
