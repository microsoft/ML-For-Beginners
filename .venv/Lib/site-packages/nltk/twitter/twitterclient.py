# Natural Language Toolkit: Twitter client
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#         Lorenzo Rubio <lrnzcig@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


"""
NLTK Twitter client

This module offers methods for collecting and processing Tweets. Most of the
functionality depends on access to the Twitter APIs, and this is handled via
the third party Twython library.

If one of the methods below returns an integer, it is probably a `Twitter
error code <https://dev.twitter.com/overview/api/response-codes>`_. For
example, the response of '420' means that you have reached the limit of the
requests you can currently make to the Twitter API. Currently, `rate limits
for the search API <https://dev.twitter.com/rest/public/rate-limiting>`_ are
divided into 15 minute windows.
"""

import datetime
import gzip
import itertools
import json
import os
import time

import requests
from twython import Twython, TwythonStreamer
from twython.exceptions import TwythonError, TwythonRateLimitError

from nltk.twitter.api import BasicTweetHandler, TweetHandlerI
from nltk.twitter.util import credsfromfile, guess_path


class Streamer(TwythonStreamer):
    """
    Retrieve data from the Twitter Streaming API.

    The streaming API requires
    `OAuth 1.0 <https://en.wikipedia.org/wiki/OAuth>`_ authentication.
    """

    def __init__(self, app_key, app_secret, oauth_token, oauth_token_secret):

        self.handler = None
        self.do_continue = True
        TwythonStreamer.__init__(
            self, app_key, app_secret, oauth_token, oauth_token_secret
        )

    def register(self, handler):
        """
        Register a method for handling Tweets.

        :param TweetHandlerI handler: method for viewing
        """
        self.handler = handler

    def on_success(self, data):
        """
        :param data: response from Twitter API
        """
        if self.do_continue:
            if self.handler is not None:
                if "text" in data:
                    self.handler.counter += 1
                    self.handler.handle(data)
                    self.do_continue = self.handler.do_continue()
            else:
                raise ValueError("No data handler has been registered.")
        else:
            self.disconnect()
            self.handler.on_finish()

    def on_error(self, status_code, data):
        """
        :param status_code: The status code returned by the Twitter API
        :param data: The response from Twitter API

        """
        print(status_code)

    def sample(self):
        """
        Wrapper for 'statuses / sample' API call
        """
        while self.do_continue:

            # Stream in an endless loop until limit is reached. See twython
            # issue 288: https://github.com/ryanmcgrath/twython/issues/288
            # colditzjb commented on 9 Dec 2014

            try:
                self.statuses.sample()
            except requests.exceptions.ChunkedEncodingError as e:
                if e is not None:
                    print(f"Error (stream will continue): {e}")
                continue

    def filter(self, track="", follow="", lang="en"):
        """
        Wrapper for 'statuses / filter' API call
        """
        while self.do_continue:
            # Stream in an endless loop until limit is reached

            try:
                if track == "" and follow == "":
                    msg = "Please supply a value for 'track', 'follow'"
                    raise ValueError(msg)
                self.statuses.filter(track=track, follow=follow, lang=lang)
            except requests.exceptions.ChunkedEncodingError as e:
                if e is not None:
                    print(f"Error (stream will continue): {e}")
                continue


class Query(Twython):
    """
    Retrieve data from the Twitter REST API.
    """

    def __init__(self, app_key, app_secret, oauth_token, oauth_token_secret):
        """
        :param app_key: (optional) Your applications key
        :param app_secret: (optional) Your applications secret key
        :param oauth_token: (optional) When using **OAuth 1**, combined with
            oauth_token_secret to make authenticated calls
        :param oauth_token_secret: (optional) When using **OAuth 1** combined
            with oauth_token to make authenticated calls
        """
        self.handler = None
        self.do_continue = True
        Twython.__init__(self, app_key, app_secret, oauth_token, oauth_token_secret)

    def register(self, handler):
        """
        Register a method for handling Tweets.

        :param TweetHandlerI handler: method for viewing or writing Tweets to a file.
        """
        self.handler = handler

    def expand_tweetids(self, ids_f, verbose=True):
        """
        Given a file object containing a list of Tweet IDs, fetch the
        corresponding full Tweets from the Twitter API.

        The API call `statuses/lookup` will fail to retrieve a Tweet if the
        user has deleted it.

        This call to the Twitter API is rate-limited. See
        <https://dev.twitter.com/rest/reference/get/statuses/lookup> for details.

        :param ids_f: input file object consisting of Tweet IDs, one to a line
        :return: iterable of Tweet objects in JSON format
        """
        ids = [line.strip() for line in ids_f if line]

        if verbose:
            print(f"Counted {len(ids)} Tweet IDs in {ids_f}.")

        # The Twitter endpoint takes lists of up to 100 ids, so we chunk the
        # ids.
        id_chunks = [ids[i : i + 100] for i in range(0, len(ids), 100)]

        chunked_tweets = (self.lookup_status(id=chunk) for chunk in id_chunks)

        return itertools.chain.from_iterable(chunked_tweets)

    def _search_tweets(self, keywords, limit=100, lang="en"):
        """
        Assumes that the handler has been informed. Fetches Tweets from
        search_tweets generator output and passses them to handler

        :param str keywords: A list of query terms to search for, written as\
        a comma-separated string.
        :param int limit: Number of Tweets to process
        :param str lang: language
        """
        while True:
            tweets = self.search_tweets(
                keywords=keywords, limit=limit, lang=lang, max_id=self.handler.max_id
            )
            for tweet in tweets:
                self.handler.handle(tweet)
            if not (self.handler.do_continue() and self.handler.repeat):
                break
        self.handler.on_finish()

    def search_tweets(
        self,
        keywords,
        limit=100,
        lang="en",
        max_id=None,
        retries_after_twython_exception=0,
    ):
        """
        Call the REST API ``'search/tweets'`` endpoint with some plausible
        defaults. See `the Twitter search documentation
        <https://dev.twitter.com/rest/public/search>`_ for more information
        about admissible search parameters.

        :param str keywords: A list of query terms to search for, written as\
        a comma-separated string
        :param int limit: Number of Tweets to process
        :param str lang: language
        :param int max_id: id of the last tweet fetched
        :param int retries_after_twython_exception: number of retries when\
        searching Tweets before raising an exception
        :rtype: python generator
        """
        if not self.handler:
            # if no handler is provided, `BasicTweetHandler` provides minimum
            # functionality for limiting the number of Tweets retrieved
            self.handler = BasicTweetHandler(limit=limit)

        count_from_query = 0
        if max_id:
            self.handler.max_id = max_id
        else:
            results = self.search(
                q=keywords, count=min(100, limit), lang=lang, result_type="recent"
            )
            count = len(results["statuses"])
            if count == 0:
                print("No Tweets available through REST API for those keywords")
                return
            count_from_query = count
            self.handler.max_id = results["statuses"][count - 1]["id"] - 1

            for result in results["statuses"]:
                yield result
                self.handler.counter += 1
                if self.handler.do_continue() == False:
                    return

        # Pagination loop: keep fetching Tweets until the desired count is
        # reached while dealing with Twitter rate limits.
        retries = 0
        while count_from_query < limit:
            try:
                mcount = min(100, limit - count_from_query)
                results = self.search(
                    q=keywords,
                    count=mcount,
                    lang=lang,
                    max_id=self.handler.max_id,
                    result_type="recent",
                )
            except TwythonRateLimitError as e:
                print(f"Waiting for 15 minutes -{e}")
                time.sleep(15 * 60)  # wait 15 minutes
                continue
            except TwythonError as e:
                print(f"Fatal error in Twython request -{e}")
                if retries_after_twython_exception == retries:
                    raise e
                retries += 1

            count = len(results["statuses"])
            if count == 0:
                print("No more Tweets available through rest api")
                return
            count_from_query += count
            # the max_id is also present in the Tweet metadata
            # results['search_metadata']['next_results'], but as part of a
            # query and difficult to fetch. This is doing the equivalent
            # (last tweet id minus one)
            self.handler.max_id = results["statuses"][count - 1]["id"] - 1

            for result in results["statuses"]:
                yield result
                self.handler.counter += 1
                if self.handler.do_continue() == False:
                    return

    def user_info_from_id(self, userids):
        """
        Convert a list of userIDs into a variety of information about the users.

        See <https://dev.twitter.com/rest/reference/get/users/show>.

        :param list userids: A list of integer strings corresponding to Twitter userIDs
        :rtype: list(json)
        """
        return [self.show_user(user_id=userid) for userid in userids]

    def user_tweets(self, screen_name, limit, include_rts="false"):
        """
        Return a collection of the most recent Tweets posted by the user

        :param str user: The user's screen name; the initial '@' symbol\
        should be omitted
        :param int limit: The number of Tweets to recover; 200 is the maximum allowed
        :param str include_rts: Whether to include statuses which have been\
        retweeted by the user; possible values are 'true' and 'false'
        """
        data = self.get_user_timeline(
            screen_name=screen_name, count=limit, include_rts=include_rts
        )
        for item in data:
            self.handler.handle(item)


class Twitter:
    """
    Wrapper class with restricted functionality and fewer options.
    """

    def __init__(self):
        self._oauth = credsfromfile()
        self.streamer = Streamer(**self._oauth)
        self.query = Query(**self._oauth)

    def tweets(
        self,
        keywords="",
        follow="",
        to_screen=True,
        stream=True,
        limit=100,
        date_limit=None,
        lang="en",
        repeat=False,
        gzip_compress=False,
    ):
        """
        Process some Tweets in a simple manner.

        :param str keywords: Keywords to use for searching or filtering
        :param list follow: UserIDs to use for filtering Tweets from the public stream
        :param bool to_screen: If `True`, display the tweet texts on the screen,\
            otherwise print to a file

        :param bool stream: If `True`, use the live public stream,\
            otherwise search past public Tweets

        :param int limit: The number of data items to process in the current\
            round of processing.

        :param tuple date_limit: The date at which to stop collecting\
            new data. This should be entered as a tuple which can serve as the\
            argument to `datetime.datetime`.\
            E.g. `date_limit=(2015, 4, 1, 12, 40)` for 12:30 pm on April 1 2015.
            Note that, in the case of streaming, this is the maximum date, i.e.\
            a date in the future; if not, it is the minimum date, i.e. a date\
            in the past

        :param str lang: language

        :param bool repeat: A flag to determine whether multiple files should\
            be written. If `True`, the length of each file will be set by the\
            value of `limit`. Use only if `to_screen` is `False`. See also
            :py:func:`handle`.

        :param gzip_compress: if `True`, output files are compressed with gzip.
        """
        if stream:
            upper_date_limit = date_limit
            lower_date_limit = None
        else:
            upper_date_limit = None
            lower_date_limit = date_limit

        if to_screen:
            handler = TweetViewer(
                limit=limit,
                upper_date_limit=upper_date_limit,
                lower_date_limit=lower_date_limit,
            )
        else:
            handler = TweetWriter(
                limit=limit,
                upper_date_limit=upper_date_limit,
                lower_date_limit=lower_date_limit,
                repeat=repeat,
                gzip_compress=gzip_compress,
            )

        if to_screen:
            handler = TweetViewer(limit=limit)
        else:
            if stream:
                upper_date_limit = date_limit
                lower_date_limit = None
            else:
                upper_date_limit = None
                lower_date_limit = date_limit

            handler = TweetWriter(
                limit=limit,
                upper_date_limit=upper_date_limit,
                lower_date_limit=lower_date_limit,
                repeat=repeat,
                gzip_compress=gzip_compress,
            )

        if stream:
            self.streamer.register(handler)
            if keywords == "" and follow == "":
                self.streamer.sample()
            else:
                self.streamer.filter(track=keywords, follow=follow, lang=lang)
        else:
            self.query.register(handler)
            if keywords == "":
                raise ValueError("Please supply at least one keyword to search for.")
            else:
                self.query._search_tweets(keywords, limit=limit, lang=lang)


class TweetViewer(TweetHandlerI):
    """
    Handle data by sending it to the terminal.
    """

    def handle(self, data):
        """
        Direct data to `sys.stdout`

        :return: return ``False`` if processing should cease, otherwise return ``True``.
        :rtype: bool
        :param data: Tweet object returned by Twitter API
        """
        text = data["text"]
        print(text)

        self.check_date_limit(data)
        if self.do_stop:
            return

    def on_finish(self):
        print(f"Written {self.counter} Tweets")


class TweetWriter(TweetHandlerI):
    """
    Handle data by writing it to a file.
    """

    def __init__(
        self,
        limit=2000,
        upper_date_limit=None,
        lower_date_limit=None,
        fprefix="tweets",
        subdir="twitter-files",
        repeat=False,
        gzip_compress=False,
    ):
        """
        The difference between the upper and lower date limits depends on
        whether Tweets are coming in an ascending date order (i.e. when
        streaming) or descending date order (i.e. when searching past Tweets).

        :param int limit: number of data items to process in the current\
        round of processing.

        :param tuple upper_date_limit: The date at which to stop collecting new\
        data. This should be entered as a tuple which can serve as the\
        argument to `datetime.datetime`. E.g. `upper_date_limit=(2015, 4, 1, 12,\
        40)` for 12:30 pm on April 1 2015.

        :param tuple lower_date_limit: The date at which to stop collecting new\
        data. See `upper_data_limit` for formatting.

        :param str fprefix: The prefix to use in creating file names for Tweet\
        collections.

        :param str subdir: The name of the directory where Tweet collection\
        files should be stored.

        :param bool repeat: flag to determine whether multiple files should be\
        written. If `True`, the length of each file will be set by the value\
        of `limit`. See also :py:func:`handle`.

        :param gzip_compress: if `True`, output files are compressed with gzip.
        """
        self.fprefix = fprefix
        self.subdir = guess_path(subdir)
        self.gzip_compress = gzip_compress
        self.fname = self.timestamped_file()
        self.repeat = repeat
        self.output = None
        TweetHandlerI.__init__(self, limit, upper_date_limit, lower_date_limit)

    def timestamped_file(self):
        """
        :return: timestamped file name
        :rtype: str
        """
        subdir = self.subdir
        fprefix = self.fprefix
        if subdir:
            if not os.path.exists(subdir):
                os.mkdir(subdir)

        fname = os.path.join(subdir, fprefix)
        fmt = "%Y%m%d-%H%M%S"
        timestamp = datetime.datetime.now().strftime(fmt)
        if self.gzip_compress:
            suffix = ".gz"
        else:
            suffix = ""
        outfile = f"{fname}.{timestamp}.json{suffix}"
        return outfile

    def handle(self, data):
        """
        Write Twitter data as line-delimited JSON into one or more files.

        :return: return `False` if processing should cease, otherwise return `True`.
        :param data: tweet object returned by Twitter API
        """
        if self.startingup:
            if self.gzip_compress:
                self.output = gzip.open(self.fname, "w")
            else:
                self.output = open(self.fname, "w")
            print(f"Writing to {self.fname}")

        json_data = json.dumps(data)
        if self.gzip_compress:
            self.output.write((json_data + "\n").encode("utf-8"))
        else:
            self.output.write(json_data + "\n")

        self.check_date_limit(data)
        if self.do_stop:
            return

        self.startingup = False

    def on_finish(self):
        print(f"Written {self.counter} Tweets")
        if self.output:
            self.output.close()

    def do_continue(self):
        if self.repeat == False:
            return TweetHandlerI.do_continue(self)

        if self.do_stop:
            # stop for a functional cause (e.g. date limit)
            return False

        if self.counter == self.limit:
            # repeat is True, thus close output file and
            # create a new one
            self._restart_file()
        return True

    def _restart_file(self):
        self.on_finish()
        self.fname = self.timestamped_file()
        self.startingup = True
        self.counter = 0
