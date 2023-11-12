"""
Tests for static parts of Twitter package
"""

import os

import pytest

pytest.importorskip("twython")

from nltk.twitter import Authenticate


@pytest.fixture
def auth():
    return Authenticate()


class TestCredentials:
    """
    Tests that Twitter credentials from a file are handled correctly.
    """

    @classmethod
    def setup_class(self):
        self.subdir = os.path.join(os.path.dirname(__file__), "files")
        os.environ["TWITTER"] = "twitter-files"

    def test_environment(self, auth):
        """
        Test that environment variable has been read correctly.
        """
        fn = os.path.basename(auth.creds_subdir)
        assert fn == os.environ["TWITTER"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            # Each of the following scenarios should raise an error:
            # An empty subdir path
            {"subdir": ""},
            # A subdir path of None
            {"subdir": None},
            # A nonexistent directory
            {"subdir": "/nosuchdir"},
            # 'credentials.txt' is not in default subdir, as read from `os.environ['TWITTER']`
            {},
            # Nonexistent credentials file ('foobar')
            {"creds_file": "foobar"},
            # 'bad_oauth1-1.txt' is incomplete
            {"creds_file": "bad_oauth1-1.txt"},
            # The first key in credentials file 'bad_oauth1-2.txt' is ill-formed
            {"creds_file": "bad_oauth1-2.txt"},
            # The first two lines in 'bad_oauth1-3.txt' are collapsed
            {"creds_file": "bad_oauth1-3.txt"},
        ],
    )
    def test_scenarios_that_should_raise_errors(self, kwargs, auth):
        """Various scenarios that should raise errors"""
        try:
            auth.load_creds(**kwargs)
        # raises ValueError (zero length field name in format) for python 2.6
        # OSError for the rest
        except (OSError, ValueError):
            pass
        except Exception as e:
            pytest.fail("Unexpected exception thrown: %s" % e)
        else:
            pytest.fail("OSError exception not thrown.")

    def test_correct_file(self, auth):
        """Test that a proper file succeeds and is read correctly"""
        oauth = auth.load_creds(subdir=self.subdir)

        assert auth.creds_fullpath == os.path.join(self.subdir, auth.creds_file)
        assert auth.creds_file == "credentials.txt"
        assert oauth["app_key"] == "a"
