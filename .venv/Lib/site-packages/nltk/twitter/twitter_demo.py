# Natural Language Toolkit: Twitter client
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#         Lorenzo Rubio <lrnzcig@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Examples to demo the :py:mod:`twitterclient` code.

These demo functions should all run, with the following caveats:

* You must have obtained API keys from Twitter, and installed them according to
  the instructions in the `twitter HOWTO <https://www.nltk.org/howto/twitter.html>`_.

* If you are on a slow network, some of the calls to the Twitter API may
  timeout.

* If you are being rate limited while searching, you will receive a 420
  error response.

* Your terminal window / console must be able to display UTF-8 encoded characters.

For documentation about the Twitter APIs, see `The Streaming APIs Overview
<https://dev.twitter.com/streaming/overview>`_ and `The REST APIs Overview
<https://dev.twitter.com/rest/public>`_.

For error codes see Twitter's
`Error Codes and Responses <https://dev.twitter.com/overview/api/response-codes>`
"""

import datetime
import json
from functools import wraps
from io import StringIO

from nltk.twitter import (
    Query,
    Streamer,
    TweetViewer,
    TweetWriter,
    Twitter,
    credsfromfile,
)

SPACER = "###################################"


def verbose(func):
    """Decorator for demo functions"""

    @wraps(func)
    def with_formatting(*args, **kwargs):
        print()
        print(SPACER)
        print("Using %s" % (func.__name__))
        print(SPACER)
        return func(*args, **kwargs)

    return with_formatting


def yesterday():
    """
    Get yesterday's datetime as a 5-tuple.
    """
    date = datetime.datetime.now()
    date -= datetime.timedelta(days=1)
    date_tuple = date.timetuple()[:6]
    return date_tuple


def setup():
    """
    Initialize global variables for the demos.
    """
    global USERIDS, FIELDS

    USERIDS = ["759251", "612473", "15108702", "6017542", "2673523800"]
    # UserIDs corresponding to\
    #           @CNN,    @BBCNews, @ReutersLive, @BreakingNews, @AJELive
    FIELDS = ["id_str"]


@verbose
def twitterclass_demo():
    """
    Use the simplified :class:`Twitter` class to write some tweets to a file.
    """
    tw = Twitter()
    print("Track from the public stream\n")
    tw.tweets(keywords="love, hate", limit=10)  # public stream
    print(SPACER)
    print("Search past Tweets\n")
    tw = Twitter()
    tw.tweets(keywords="love, hate", stream=False, limit=10)  # search past tweets
    print(SPACER)
    print(
        "Follow two accounts in the public stream"
        + " -- be prepared to wait a few minutes\n"
    )
    tw = Twitter()
    tw.tweets(follow=["759251", "6017542"], stream=True, limit=5)  # public stream


@verbose
def sampletoscreen_demo(limit=20):
    """
    Sample from the Streaming API and send output to terminal.
    """
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.sample()


@verbose
def tracktoscreen_demo(track="taylor swift", limit=10):
    """
    Track keywords from the public Streaming API and send output to terminal.
    """
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.filter(track=track)


@verbose
def search_demo(keywords="nltk"):
    """
    Use the REST API to search for past tweets containing a given keyword.
    """
    oauth = credsfromfile()
    client = Query(**oauth)
    for tweet in client.search_tweets(keywords=keywords, limit=10):
        print(tweet["text"])


@verbose
def tweets_by_user_demo(user="NLTK_org", count=200):
    """
    Use the REST API to search for past tweets by a given user.
    """
    oauth = credsfromfile()
    client = Query(**oauth)
    client.register(TweetWriter())
    client.user_tweets(user, count)


@verbose
def lookup_by_userid_demo():
    """
    Use the REST API to convert a userID to a screen name.
    """
    oauth = credsfromfile()
    client = Query(**oauth)
    user_info = client.user_info_from_id(USERIDS)
    for info in user_info:
        name = info["screen_name"]
        followers = info["followers_count"]
        following = info["friends_count"]
        print(f"{name}, followers: {followers}, following: {following}")


@verbose
def followtoscreen_demo(limit=10):
    """
    Using the Streaming API, select just the tweets from a specified list of
    userIDs.

    This is will only give results in a reasonable time if the users in
    question produce a high volume of tweets, and may even so show some delay.
    """
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetViewer(limit=limit))
    client.statuses.filter(follow=USERIDS)


@verbose
def streamtofile_demo(limit=20):
    """
    Write 20 tweets sampled from the public Streaming API to a file.
    """
    oauth = credsfromfile()
    client = Streamer(**oauth)
    client.register(TweetWriter(limit=limit, repeat=False))
    client.statuses.sample()


@verbose
def limit_by_time_demo(keywords="nltk"):
    """
    Query the REST API for Tweets about NLTK since yesterday and send
    the output to terminal.

    This example makes the assumption that there are sufficient Tweets since
    yesterday for the date to be an effective cut-off.
    """
    date = yesterday()
    dt_date = datetime.datetime(*date)
    oauth = credsfromfile()
    client = Query(**oauth)
    client.register(TweetViewer(limit=100, lower_date_limit=date))

    print(f"Cutoff date: {dt_date}\n")

    for tweet in client.search_tweets(keywords=keywords):
        print("{} ".format(tweet["created_at"]), end="")
        client.handler.handle(tweet)


@verbose
def corpusreader_demo():
    """
    Use `TwitterCorpusReader` tp read a file of tweets, and print out

    * some full tweets in JSON format;
    * some raw strings from the tweets (i.e., the value of the `text` field); and
    * the result of tokenising the raw strings.

    """
    from nltk.corpus import twitter_samples as tweets

    print()
    print("Complete tweet documents")
    print(SPACER)
    for tweet in tweets.docs("tweets.20150430-223406.json")[:1]:
        print(json.dumps(tweet, indent=1, sort_keys=True))

    print()
    print("Raw tweet strings:")
    print(SPACER)
    for text in tweets.strings("tweets.20150430-223406.json")[:15]:
        print(text)

    print()
    print("Tokenized tweet strings:")
    print(SPACER)
    for toks in tweets.tokenized("tweets.20150430-223406.json")[:15]:
        print(toks)


@verbose
def expand_tweetids_demo():
    """
    Given a file object containing a list of Tweet IDs, fetch the
    corresponding full Tweets, if available.

    """
    ids_f = StringIO(
        """\
        588665495492124672
        588665495487909888
        588665495508766721
        588665495513006080
        588665495517200384
        588665495487811584
        588665495525588992
        588665495487844352
        588665495492014081
        588665495512948737"""
    )
    oauth = credsfromfile()
    client = Query(**oauth)
    hydrated = client.expand_tweetids(ids_f)

    for tweet in hydrated:
        id_str = tweet["id_str"]
        print(f"id: {id_str}")
        text = tweet["text"]
        if text.startswith("@null"):
            text = "[Tweet not available]"
        print(text + "\n")


ALL = [
    twitterclass_demo,
    sampletoscreen_demo,
    tracktoscreen_demo,
    search_demo,
    tweets_by_user_demo,
    lookup_by_userid_demo,
    followtoscreen_demo,
    streamtofile_demo,
    limit_by_time_demo,
    corpusreader_demo,
    expand_tweetids_demo,
]

"""
Select demo functions to run. E.g. replace the following line with "DEMOS =
ALL[8:]" to execute only the final three demos.
"""
DEMOS = ALL[:]

if __name__ == "__main__":
    setup()

    for demo in DEMOS:
        demo()

    print("\n" + SPACER)
    print("All demos completed")
    print(SPACER)
