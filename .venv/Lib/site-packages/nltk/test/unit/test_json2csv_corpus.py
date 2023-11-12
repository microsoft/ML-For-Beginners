# Natural Language Toolkit: Twitter client
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Lorenzo Rubio <lrnzcig@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Regression tests for `json2csv()` and `json2csv_entities()` in Twitter
package.
"""
from pathlib import Path

import pytest

from nltk.corpus import twitter_samples
from nltk.twitter.common import json2csv, json2csv_entities


def files_are_identical(pathA, pathB):
    """
    Compare two files, ignoring carriage returns,
    leading whitespace, and trailing whitespace
    """
    f1 = [l.strip() for l in pathA.read_bytes().splitlines()]
    f2 = [l.strip() for l in pathB.read_bytes().splitlines()]
    return f1 == f2


subdir = Path(__file__).parent / "files"


@pytest.fixture
def infile():
    with open(twitter_samples.abspath("tweets.20150430-223406.json")) as infile:
        return [next(infile) for x in range(100)]


def test_textoutput(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.text.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.text.csv"
    json2csv(infile, outfn, ["text"], gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)


def test_tweet_metadata(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.tweet.csv.ref"
    fields = [
        "created_at",
        "favorite_count",
        "id",
        "in_reply_to_status_id",
        "in_reply_to_user_id",
        "retweet_count",
        "retweeted",
        "text",
        "truncated",
        "user.id",
    ]

    outfn = tmp_path / "tweets.20150430-223406.tweet.csv"
    json2csv(infile, outfn, fields, gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)


def test_user_metadata(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.user.csv.ref"
    fields = ["id", "text", "user.id", "user.followers_count", "user.friends_count"]

    outfn = tmp_path / "tweets.20150430-223406.user.csv"
    json2csv(infile, outfn, fields, gzip_compress=False)
    assert files_are_identical(outfn, ref_fn)


def test_tweet_hashtag(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.hashtag.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.hashtag.csv"
    json2csv_entities(
        infile,
        outfn,
        ["id", "text"],
        "hashtags",
        ["text"],
        gzip_compress=False,
    )
    assert files_are_identical(outfn, ref_fn)


def test_tweet_usermention(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.usermention.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.usermention.csv"
    json2csv_entities(
        infile,
        outfn,
        ["id", "text"],
        "user_mentions",
        ["id", "screen_name"],
        gzip_compress=False,
    )
    assert files_are_identical(outfn, ref_fn)


def test_tweet_media(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.media.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.media.csv"
    json2csv_entities(
        infile,
        outfn,
        ["id"],
        "media",
        ["media_url", "url"],
        gzip_compress=False,
    )

    assert files_are_identical(outfn, ref_fn)


def test_tweet_url(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.url.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.url.csv"
    json2csv_entities(
        infile,
        outfn,
        ["id"],
        "urls",
        ["url", "expanded_url"],
        gzip_compress=False,
    )

    assert files_are_identical(outfn, ref_fn)


def test_userurl(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.userurl.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.userurl.csv"
    json2csv_entities(
        infile,
        outfn,
        ["id", "screen_name"],
        "user.urls",
        ["url", "expanded_url"],
        gzip_compress=False,
    )

    assert files_are_identical(outfn, ref_fn)


def test_tweet_place(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.place.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.place.csv"
    json2csv_entities(
        infile,
        outfn,
        ["id", "text"],
        "place",
        ["name", "country"],
        gzip_compress=False,
    )

    assert files_are_identical(outfn, ref_fn)


def test_tweet_place_boundingbox(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.placeboundingbox.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.placeboundingbox.csv"
    json2csv_entities(
        infile,
        outfn,
        ["id", "name"],
        "place.bounding_box",
        ["coordinates"],
        gzip_compress=False,
    )

    assert files_are_identical(outfn, ref_fn)


def test_retweet_original_tweet(tmp_path, infile):
    ref_fn = subdir / "tweets.20150430-223406.retweet.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.retweet.csv"
    json2csv_entities(
        infile,
        outfn,
        ["id"],
        "retweeted_status",
        [
            "created_at",
            "favorite_count",
            "id",
            "in_reply_to_status_id",
            "in_reply_to_user_id",
            "retweet_count",
            "text",
            "truncated",
            "user.id",
        ],
        gzip_compress=False,
    )

    assert files_are_identical(outfn, ref_fn)


def test_file_is_wrong(tmp_path, infile):
    """
    Sanity check that file comparison is not giving false positives.
    """
    ref_fn = subdir / "tweets.20150430-223406.retweet.csv.ref"
    outfn = tmp_path / "tweets.20150430-223406.text.csv"
    json2csv(infile, outfn, ["text"], gzip_compress=False)
    assert not files_are_identical(outfn, ref_fn)
