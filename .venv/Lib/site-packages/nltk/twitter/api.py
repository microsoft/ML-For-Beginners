# Natural Language Toolkit: Twitter API
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#         Lorenzo Rubio <lrnzcig@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
This module provides an interface for TweetHandlers, and support for timezone
handling.
"""

import time as _time
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta, timezone, tzinfo


class LocalTimezoneOffsetWithUTC(tzinfo):
    """
    This is not intended to be a general purpose class for dealing with the
    local timezone. In particular:

    * it assumes that the date passed has been created using
      `datetime(..., tzinfo=Local)`, where `Local` is an instance of
      the object `LocalTimezoneOffsetWithUTC`;
    * for such an object, it returns the offset with UTC, used for date comparisons.

    Reference: https://docs.python.org/3/library/datetime.html
    """

    STDOFFSET = timedelta(seconds=-_time.timezone)

    if _time.daylight:
        DSTOFFSET = timedelta(seconds=-_time.altzone)
    else:
        DSTOFFSET = STDOFFSET

    def utcoffset(self, dt):
        """
        Access the relevant time offset.
        """
        return self.DSTOFFSET


LOCAL = LocalTimezoneOffsetWithUTC()


class BasicTweetHandler(metaclass=ABCMeta):
    """
    Minimal implementation of `TweetHandler`.

    Counts the number of Tweets and decides when the client should stop
    fetching them.
    """

    def __init__(self, limit=20):
        self.limit = limit
        self.counter = 0

        """
        A flag to indicate to the client whether to stop fetching data given
        some condition (e.g., reaching a date limit).
        """
        self.do_stop = False

        """
        Stores the id of the last fetched Tweet to handle pagination.
        """
        self.max_id = None

    def do_continue(self):
        """
        Returns `False` if the client should stop fetching Tweets.
        """
        return self.counter < self.limit and not self.do_stop


class TweetHandlerI(BasicTweetHandler):
    """
    Interface class whose subclasses should implement a handle method that
    Twitter clients can delegate to.
    """

    def __init__(self, limit=20, upper_date_limit=None, lower_date_limit=None):
        """
        :param int limit: The number of data items to process in the current\
        round of processing.

        :param tuple upper_date_limit: The date at which to stop collecting\
        new data. This should be entered as a tuple which can serve as the\
        argument to `datetime.datetime`.\
        E.g. `date_limit=(2015, 4, 1, 12, 40)` for 12:30 pm on April 1 2015.

        :param tuple lower_date_limit: The date at which to stop collecting\
        new data. See `upper_data_limit` for formatting.
        """
        BasicTweetHandler.__init__(self, limit)

        self.upper_date_limit = None
        self.lower_date_limit = None
        if upper_date_limit:
            self.upper_date_limit = datetime(*upper_date_limit, tzinfo=LOCAL)
        if lower_date_limit:
            self.lower_date_limit = datetime(*lower_date_limit, tzinfo=LOCAL)

        self.startingup = True

    @abstractmethod
    def handle(self, data):
        """
        Deal appropriately with data returned by the Twitter API
        """

    @abstractmethod
    def on_finish(self):
        """
        Actions when the tweet limit has been reached
        """

    def check_date_limit(self, data, verbose=False):
        """
        Validate date limits.
        """
        if self.upper_date_limit or self.lower_date_limit:
            date_fmt = "%a %b %d %H:%M:%S +0000 %Y"
            tweet_date = datetime.strptime(data["created_at"], date_fmt).replace(
                tzinfo=timezone.utc
            )
            if (self.upper_date_limit and tweet_date > self.upper_date_limit) or (
                self.lower_date_limit and tweet_date < self.lower_date_limit
            ):
                if self.upper_date_limit:
                    message = "earlier"
                    date_limit = self.upper_date_limit
                else:
                    message = "later"
                    date_limit = self.lower_date_limit
                if verbose:
                    print(
                        "Date limit {} is {} than date of current tweet {}".format(
                            date_limit, message, tweet_date
                        )
                    )
                self.do_stop = True
