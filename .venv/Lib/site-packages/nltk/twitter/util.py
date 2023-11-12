# Natural Language Toolkit: Twitter client
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
#         Lorenzo Rubio <lrnzcig@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Authentication utilities to accompany `twitterclient`.
"""

import os
import pprint

from twython import Twython


def credsfromfile(creds_file=None, subdir=None, verbose=False):
    """
    Convenience function for authentication
    """
    return Authenticate().load_creds(
        creds_file=creds_file, subdir=subdir, verbose=verbose
    )


class Authenticate:
    """
    Methods for authenticating with Twitter.
    """

    def __init__(self):
        self.creds_file = "credentials.txt"
        self.creds_fullpath = None

        self.oauth = {}
        try:
            self.twitter_dir = os.environ["TWITTER"]
            self.creds_subdir = self.twitter_dir
        except KeyError:
            self.twitter_dir = None
            self.creds_subdir = None

    def load_creds(self, creds_file=None, subdir=None, verbose=False):
        """
        Read OAuth credentials from a text file.

        File format for OAuth 1::

           app_key=YOUR_APP_KEY
           app_secret=YOUR_APP_SECRET
           oauth_token=OAUTH_TOKEN
           oauth_token_secret=OAUTH_TOKEN_SECRET


        File format for OAuth 2::

           app_key=YOUR_APP_KEY
           app_secret=YOUR_APP_SECRET
           access_token=ACCESS_TOKEN

        :param str file_name: File containing credentials. ``None`` (default) reads
            data from `TWITTER/'credentials.txt'`
        """
        if creds_file is not None:
            self.creds_file = creds_file

        if subdir is None:
            if self.creds_subdir is None:
                msg = (
                    "Supply a value to the 'subdir' parameter or"
                    + " set the TWITTER environment variable."
                )
                raise ValueError(msg)
        else:
            self.creds_subdir = subdir

        self.creds_fullpath = os.path.normpath(
            os.path.join(self.creds_subdir, self.creds_file)
        )

        if not os.path.isfile(self.creds_fullpath):
            raise OSError(f"Cannot find file {self.creds_fullpath}")

        with open(self.creds_fullpath) as infile:
            if verbose:
                print(f"Reading credentials file {self.creds_fullpath}")

            for line in infile:
                if "=" in line:
                    name, value = line.split("=", 1)
                    self.oauth[name.strip()] = value.strip()

        self._validate_creds_file(verbose=verbose)

        return self.oauth

    def _validate_creds_file(self, verbose=False):
        """Check validity of a credentials file."""
        oauth1 = False
        oauth1_keys = ["app_key", "app_secret", "oauth_token", "oauth_token_secret"]
        oauth2 = False
        oauth2_keys = ["app_key", "app_secret", "access_token"]
        if all(k in self.oauth for k in oauth1_keys):
            oauth1 = True
        elif all(k in self.oauth for k in oauth2_keys):
            oauth2 = True

        if not (oauth1 or oauth2):
            msg = f"Missing or incorrect entries in {self.creds_file}\n"
            msg += pprint.pformat(self.oauth)
            raise ValueError(msg)
        elif verbose:
            print(f'Credentials file "{self.creds_file}" looks good')


def add_access_token(creds_file=None):
    """
    For OAuth 2, retrieve an access token for an app and append it to a
    credentials file.
    """
    if creds_file is None:
        path = os.path.dirname(__file__)
        creds_file = os.path.join(path, "credentials2.txt")
    oauth2 = credsfromfile(creds_file=creds_file)
    app_key = oauth2["app_key"]
    app_secret = oauth2["app_secret"]

    twitter = Twython(app_key, app_secret, oauth_version=2)
    access_token = twitter.obtain_access_token()
    tok = f"access_token={access_token}\n"
    with open(creds_file, "a") as infile:
        print(tok, file=infile)


def guess_path(pth):
    """
    If the path is not absolute, guess that it is a subdirectory of the
    user's home directory.

    :param str pth: The pathname of the directory where files of tweets should be written
    """
    if os.path.isabs(pth):
        return pth
    else:
        return os.path.expanduser(os.path.join("~", pth))
