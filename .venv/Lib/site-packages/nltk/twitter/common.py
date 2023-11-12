# Natural Language Toolkit: Twitter client
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#         Lorenzo Rubio <lrnzcig@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Utility functions for the `twitterclient` module which do not require
the `twython` library to have been installed.
"""
import csv
import gzip
import json

from nltk.internals import deprecated

HIER_SEPARATOR = "."


def extract_fields(tweet, fields):
    """
    Extract field values from a full tweet and return them as a list

    :param json tweet: The tweet in JSON format
    :param list fields: The fields to be extracted from the tweet
    :rtype: list(str)
    """
    out = []
    for field in fields:
        try:
            _add_field_to_out(tweet, field, out)
        except TypeError as e:
            raise RuntimeError(
                "Fatal error when extracting fields. Cannot find field ", field
            ) from e
    return out


def _add_field_to_out(json, field, out):
    if _is_composed_key(field):
        key, value = _get_key_value_composed(field)
        _add_field_to_out(json[key], value, out)
    else:
        out += [json[field]]


def _is_composed_key(field):
    return HIER_SEPARATOR in field


def _get_key_value_composed(field):
    out = field.split(HIER_SEPARATOR)
    # there could be up to 3 levels
    key = out[0]
    value = HIER_SEPARATOR.join(out[1:])
    return key, value


def _get_entity_recursive(json, entity):
    if not json:
        return None
    elif isinstance(json, dict):
        for key, value in json.items():
            if key == entity:
                return value
            # 'entities' and 'extended_entities' are wrappers in Twitter json
            # structure that contain other Twitter objects. See:
            # https://dev.twitter.com/overview/api/entities-in-twitter-objects

            if key == "entities" or key == "extended_entities":
                candidate = _get_entity_recursive(value, entity)
                if candidate is not None:
                    return candidate
        return None
    elif isinstance(json, list):
        for item in json:
            candidate = _get_entity_recursive(item, entity)
            if candidate is not None:
                return candidate
        return None
    else:
        return None


def json2csv(
    fp, outfile, fields, encoding="utf8", errors="replace", gzip_compress=False
):
    """
    Extract selected fields from a file of line-separated JSON tweets and
    write to a file in CSV format.

    This utility function allows a file of full tweets to be easily converted
    to a CSV file for easier processing. For example, just TweetIDs or
    just the text content of the Tweets can be extracted.

    Additionally, the function allows combinations of fields of other Twitter
    objects (mainly the users, see below).

    For Twitter entities (e.g. hashtags of a Tweet), and for geolocation, see
    `json2csv_entities`

    :param str infile: The name of the file containing full tweets

    :param str outfile: The name of the text file where results should be\
    written

    :param list fields: The list of fields to be extracted. Useful examples\
    are 'id_str' for the tweetID and 'text' for the text of the tweet. See\
    <https://dev.twitter.com/overview/api/tweets> for a full list of fields.\
    e. g.: ['id_str'], ['id', 'text', 'favorite_count', 'retweet_count']\
    Additionally, it allows IDs from other Twitter objects, e. g.,\
    ['id', 'text', 'user.id', 'user.followers_count', 'user.friends_count']

    :param error: Behaviour for encoding errors, see\
    https://docs.python.org/3/library/codecs.html#codec-base-classes

    :param gzip_compress: if `True`, output files are compressed with gzip
    """
    (writer, outf) = _outf_writer(outfile, encoding, errors, gzip_compress)
    # write the list of fields as header
    writer.writerow(fields)
    # process the file
    for line in fp:
        tweet = json.loads(line)
        row = extract_fields(tweet, fields)
        writer.writerow(row)
    outf.close()


@deprecated("Use open() and csv.writer() directly instead.")
def outf_writer_compat(outfile, encoding, errors, gzip_compress=False):
    """Get a CSV writer with optional compression."""
    return _outf_writer(outfile, encoding, errors, gzip_compress)


def _outf_writer(outfile, encoding, errors, gzip_compress=False):
    if gzip_compress:
        outf = gzip.open(outfile, "wt", newline="", encoding=encoding, errors=errors)
    else:
        outf = open(outfile, "w", newline="", encoding=encoding, errors=errors)
    writer = csv.writer(outf)
    return (writer, outf)


def json2csv_entities(
    tweets_file,
    outfile,
    main_fields,
    entity_type,
    entity_fields,
    encoding="utf8",
    errors="replace",
    gzip_compress=False,
):
    """
    Extract selected fields from a file of line-separated JSON tweets and
    write to a file in CSV format.

    This utility function allows a file of full Tweets to be easily converted
    to a CSV file for easier processing of Twitter entities. For example, the
    hashtags or media elements of a tweet can be extracted.

    It returns one line per entity of a Tweet, e.g. if a tweet has two hashtags
    there will be two lines in the output file, one per hashtag

    :param tweets_file: the file-like object containing full Tweets

    :param str outfile: The path of the text file where results should be\
        written

    :param list main_fields: The list of fields to be extracted from the main\
        object, usually the tweet. Useful examples: 'id_str' for the tweetID. See\
        <https://dev.twitter.com/overview/api/tweets> for a full list of fields.
        e. g.: ['id_str'], ['id', 'text', 'favorite_count', 'retweet_count']
        If `entity_type` is expressed with hierarchy, then it is the list of\
        fields of the object that corresponds to the key of the entity_type,\
        (e.g., for entity_type='user.urls', the fields in the main_fields list\
        belong to the user object; for entity_type='place.bounding_box', the\
        files in the main_field list belong to the place object of the tweet).

    :param list entity_type: The name of the entity: 'hashtags', 'media',\
        'urls' and 'user_mentions' for the tweet object. For a user object,\
        this needs to be expressed with a hierarchy: `'user.urls'`. For the\
        bounding box of the Tweet location, use `'place.bounding_box'`.

    :param list entity_fields: The list of fields to be extracted from the\
        entity. E.g. `['text']` (of the Tweet)

    :param error: Behaviour for encoding errors, see\
        https://docs.python.org/3/library/codecs.html#codec-base-classes

    :param gzip_compress: if `True`, output files are compressed with gzip
    """

    (writer, outf) = _outf_writer(outfile, encoding, errors, gzip_compress)
    header = get_header_field_list(main_fields, entity_type, entity_fields)
    writer.writerow(header)
    for line in tweets_file:
        tweet = json.loads(line)
        if _is_composed_key(entity_type):
            key, value = _get_key_value_composed(entity_type)
            object_json = _get_entity_recursive(tweet, key)
            if not object_json:
                # this can happen in the case of "place"
                continue
            object_fields = extract_fields(object_json, main_fields)
            items = _get_entity_recursive(object_json, value)
            _write_to_file(object_fields, items, entity_fields, writer)
        else:
            tweet_fields = extract_fields(tweet, main_fields)
            items = _get_entity_recursive(tweet, entity_type)
            _write_to_file(tweet_fields, items, entity_fields, writer)
    outf.close()


def get_header_field_list(main_fields, entity_type, entity_fields):
    if _is_composed_key(entity_type):
        key, value = _get_key_value_composed(entity_type)
        main_entity = key
        sub_entity = value
    else:
        main_entity = None
        sub_entity = entity_type

    if main_entity:
        output1 = [HIER_SEPARATOR.join([main_entity, x]) for x in main_fields]
    else:
        output1 = main_fields
    output2 = [HIER_SEPARATOR.join([sub_entity, x]) for x in entity_fields]
    return output1 + output2


def _write_to_file(object_fields, items, entity_fields, writer):
    if not items:
        # it could be that the entity is just not present for the tweet
        # e.g. tweet hashtag is always present, even as [], however
        # tweet media may not be present
        return
    if isinstance(items, dict):
        # this happens e.g. for "place" of a tweet
        row = object_fields
        # there might be composed keys in de list of required fields
        entity_field_values = [x for x in entity_fields if not _is_composed_key(x)]
        entity_field_composed = [x for x in entity_fields if _is_composed_key(x)]
        for field in entity_field_values:
            value = items[field]
            if isinstance(value, list):
                row += value
            else:
                row += [value]
        # now check required dictionaries
        for d in entity_field_composed:
            kd, vd = _get_key_value_composed(d)
            json_dict = items[kd]
            if not isinstance(json_dict, dict):
                raise RuntimeError(
                    """Key {} does not contain a dictionary
                in the json file""".format(
                        kd
                    )
                )
            row += [json_dict[vd]]
        writer.writerow(row)
        return
    # in general it is a list
    for item in items:
        row = object_fields + extract_fields(item, entity_fields)
        writer.writerow(row)
