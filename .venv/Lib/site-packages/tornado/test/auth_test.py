# These tests do not currently do much to verify the correct implementation
# of the openid/oauth protocols, they just exercise the major code paths
# and ensure that it doesn't blow up (e.g. with unicode/bytes issues in
# python 3)

import unittest

from tornado.auth import (
    OpenIdMixin,
    OAuthMixin,
    OAuth2Mixin,
    GoogleOAuth2Mixin,
    FacebookGraphMixin,
    TwitterMixin,
)
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError

try:
    from unittest import mock
except ImportError:
    mock = None  # type: ignore


class OpenIdClientLoginHandler(RequestHandler, OpenIdMixin):
    def initialize(self, test):
        self._OPENID_ENDPOINT = test.get_url("/openid/server/authenticate")

    @gen.coroutine
    def get(self):
        if self.get_argument("openid.mode", None):
            user = yield self.get_authenticated_user(
                http_client=self.settings["http_client"]
            )
            if user is None:
                raise Exception("user is None")
            self.finish(user)
            return
        res = self.authenticate_redirect()  # type: ignore
        assert res is None


class OpenIdServerAuthenticateHandler(RequestHandler):
    def post(self):
        if self.get_argument("openid.mode") != "check_authentication":
            raise Exception("incorrect openid.mode %r")
        self.write("is_valid:true")


class OAuth1ClientLoginHandler(RequestHandler, OAuthMixin):
    def initialize(self, test, version):
        self._OAUTH_VERSION = version
        self._OAUTH_REQUEST_TOKEN_URL = test.get_url("/oauth1/server/request_token")
        self._OAUTH_AUTHORIZE_URL = test.get_url("/oauth1/server/authorize")
        self._OAUTH_ACCESS_TOKEN_URL = test.get_url("/oauth1/server/access_token")

    def _oauth_consumer_token(self):
        return dict(key="asdf", secret="qwer")

    @gen.coroutine
    def get(self):
        if self.get_argument("oauth_token", None):
            user = yield self.get_authenticated_user(
                http_client=self.settings["http_client"]
            )
            if user is None:
                raise Exception("user is None")
            self.finish(user)
            return
        yield self.authorize_redirect(http_client=self.settings["http_client"])

    @gen.coroutine
    def _oauth_get_user_future(self, access_token):
        if self.get_argument("fail_in_get_user", None):
            raise Exception("failing in get_user")
        if access_token != dict(key="uiop", secret="5678"):
            raise Exception("incorrect access token %r" % access_token)
        return dict(email="foo@example.com")


class OAuth1ClientLoginCoroutineHandler(OAuth1ClientLoginHandler):
    """Replaces OAuth1ClientLoginCoroutineHandler's get() with a coroutine."""

    @gen.coroutine
    def get(self):
        if self.get_argument("oauth_token", None):
            # Ensure that any exceptions are set on the returned Future,
            # not simply thrown into the surrounding StackContext.
            try:
                yield self.get_authenticated_user()
            except Exception as e:
                self.set_status(503)
                self.write("got exception: %s" % e)
        else:
            yield self.authorize_redirect()


class OAuth1ClientRequestParametersHandler(RequestHandler, OAuthMixin):
    def initialize(self, version):
        self._OAUTH_VERSION = version

    def _oauth_consumer_token(self):
        return dict(key="asdf", secret="qwer")

    def get(self):
        params = self._oauth_request_parameters(
            "http://www.example.com/api/asdf",
            dict(key="uiop", secret="5678"),
            parameters=dict(foo="bar"),
        )
        self.write(params)


class OAuth1ServerRequestTokenHandler(RequestHandler):
    def get(self):
        self.write("oauth_token=zxcv&oauth_token_secret=1234")


class OAuth1ServerAccessTokenHandler(RequestHandler):
    def get(self):
        self.write("oauth_token=uiop&oauth_token_secret=5678")


class OAuth2ClientLoginHandler(RequestHandler, OAuth2Mixin):
    def initialize(self, test):
        self._OAUTH_AUTHORIZE_URL = test.get_url("/oauth2/server/authorize")

    def get(self):
        res = self.authorize_redirect()  # type: ignore
        assert res is None


class FacebookClientLoginHandler(RequestHandler, FacebookGraphMixin):
    def initialize(self, test):
        self._OAUTH_AUTHORIZE_URL = test.get_url("/facebook/server/authorize")
        self._OAUTH_ACCESS_TOKEN_URL = test.get_url("/facebook/server/access_token")
        self._FACEBOOK_BASE_URL = test.get_url("/facebook/server")

    @gen.coroutine
    def get(self):
        if self.get_argument("code", None):
            user = yield self.get_authenticated_user(
                redirect_uri=self.request.full_url(),
                client_id=self.settings["facebook_api_key"],
                client_secret=self.settings["facebook_secret"],
                code=self.get_argument("code"),
            )
            self.write(user)
        else:
            self.authorize_redirect(
                redirect_uri=self.request.full_url(),
                client_id=self.settings["facebook_api_key"],
                extra_params={"scope": "read_stream,offline_access"},
            )


class FacebookServerAccessTokenHandler(RequestHandler):
    def get(self):
        self.write(dict(access_token="asdf", expires_in=3600))


class FacebookServerMeHandler(RequestHandler):
    def get(self):
        self.write("{}")


class TwitterClientHandler(RequestHandler, TwitterMixin):
    def initialize(self, test):
        self._OAUTH_REQUEST_TOKEN_URL = test.get_url("/oauth1/server/request_token")
        self._OAUTH_ACCESS_TOKEN_URL = test.get_url("/twitter/server/access_token")
        self._OAUTH_AUTHORIZE_URL = test.get_url("/oauth1/server/authorize")
        self._OAUTH_AUTHENTICATE_URL = test.get_url("/twitter/server/authenticate")
        self._TWITTER_BASE_URL = test.get_url("/twitter/api")

    def get_auth_http_client(self):
        return self.settings["http_client"]


class TwitterClientLoginHandler(TwitterClientHandler):
    @gen.coroutine
    def get(self):
        if self.get_argument("oauth_token", None):
            user = yield self.get_authenticated_user()
            if user is None:
                raise Exception("user is None")
            self.finish(user)
            return
        yield self.authorize_redirect()


class TwitterClientAuthenticateHandler(TwitterClientHandler):
    # Like TwitterClientLoginHandler, but uses authenticate_redirect
    # instead of authorize_redirect.
    @gen.coroutine
    def get(self):
        if self.get_argument("oauth_token", None):
            user = yield self.get_authenticated_user()
            if user is None:
                raise Exception("user is None")
            self.finish(user)
            return
        yield self.authenticate_redirect()


class TwitterClientLoginGenCoroutineHandler(TwitterClientHandler):
    @gen.coroutine
    def get(self):
        if self.get_argument("oauth_token", None):
            user = yield self.get_authenticated_user()
            self.finish(user)
        else:
            # New style: with @gen.coroutine the result must be yielded
            # or else the request will be auto-finished too soon.
            yield self.authorize_redirect()


class TwitterClientShowUserHandler(TwitterClientHandler):
    @gen.coroutine
    def get(self):
        # TODO: would be nice to go through the login flow instead of
        # cheating with a hard-coded access token.
        try:
            response = yield self.twitter_request(
                "/users/show/%s" % self.get_argument("name"),
                access_token=dict(key="hjkl", secret="vbnm"),
            )
        except HTTPClientError:
            # TODO(bdarnell): Should we catch HTTP errors and
            # transform some of them (like 403s) into AuthError?
            self.set_status(500)
            self.finish("error from twitter request")
        else:
            self.finish(response)


class TwitterServerAccessTokenHandler(RequestHandler):
    def get(self):
        self.write("oauth_token=hjkl&oauth_token_secret=vbnm&screen_name=foo")


class TwitterServerShowUserHandler(RequestHandler):
    def get(self, screen_name):
        if screen_name == "error":
            raise HTTPError(500)
        assert "oauth_nonce" in self.request.arguments
        assert "oauth_timestamp" in self.request.arguments
        assert "oauth_signature" in self.request.arguments
        assert self.get_argument("oauth_consumer_key") == "test_twitter_consumer_key"
        assert self.get_argument("oauth_signature_method") == "HMAC-SHA1"
        assert self.get_argument("oauth_version") == "1.0"
        assert self.get_argument("oauth_token") == "hjkl"
        self.write(dict(screen_name=screen_name, name=screen_name.capitalize()))


class TwitterServerVerifyCredentialsHandler(RequestHandler):
    def get(self):
        assert "oauth_nonce" in self.request.arguments
        assert "oauth_timestamp" in self.request.arguments
        assert "oauth_signature" in self.request.arguments
        assert self.get_argument("oauth_consumer_key") == "test_twitter_consumer_key"
        assert self.get_argument("oauth_signature_method") == "HMAC-SHA1"
        assert self.get_argument("oauth_version") == "1.0"
        assert self.get_argument("oauth_token") == "hjkl"
        self.write(dict(screen_name="foo", name="Foo"))


class AuthTest(AsyncHTTPTestCase):
    def get_app(self):
        return Application(
            [
                # test endpoints
                ("/openid/client/login", OpenIdClientLoginHandler, dict(test=self)),
                (
                    "/oauth10/client/login",
                    OAuth1ClientLoginHandler,
                    dict(test=self, version="1.0"),
                ),
                (
                    "/oauth10/client/request_params",
                    OAuth1ClientRequestParametersHandler,
                    dict(version="1.0"),
                ),
                (
                    "/oauth10a/client/login",
                    OAuth1ClientLoginHandler,
                    dict(test=self, version="1.0a"),
                ),
                (
                    "/oauth10a/client/login_coroutine",
                    OAuth1ClientLoginCoroutineHandler,
                    dict(test=self, version="1.0a"),
                ),
                (
                    "/oauth10a/client/request_params",
                    OAuth1ClientRequestParametersHandler,
                    dict(version="1.0a"),
                ),
                ("/oauth2/client/login", OAuth2ClientLoginHandler, dict(test=self)),
                ("/facebook/client/login", FacebookClientLoginHandler, dict(test=self)),
                ("/twitter/client/login", TwitterClientLoginHandler, dict(test=self)),
                (
                    "/twitter/client/authenticate",
                    TwitterClientAuthenticateHandler,
                    dict(test=self),
                ),
                (
                    "/twitter/client/login_gen_coroutine",
                    TwitterClientLoginGenCoroutineHandler,
                    dict(test=self),
                ),
                (
                    "/twitter/client/show_user",
                    TwitterClientShowUserHandler,
                    dict(test=self),
                ),
                # simulated servers
                ("/openid/server/authenticate", OpenIdServerAuthenticateHandler),
                ("/oauth1/server/request_token", OAuth1ServerRequestTokenHandler),
                ("/oauth1/server/access_token", OAuth1ServerAccessTokenHandler),
                ("/facebook/server/access_token", FacebookServerAccessTokenHandler),
                ("/facebook/server/me", FacebookServerMeHandler),
                ("/twitter/server/access_token", TwitterServerAccessTokenHandler),
                (r"/twitter/api/users/show/(.*)\.json", TwitterServerShowUserHandler),
                (
                    r"/twitter/api/account/verify_credentials\.json",
                    TwitterServerVerifyCredentialsHandler,
                ),
            ],
            http_client=self.http_client,
            twitter_consumer_key="test_twitter_consumer_key",
            twitter_consumer_secret="test_twitter_consumer_secret",
            facebook_api_key="test_facebook_api_key",
            facebook_secret="test_facebook_secret",
        )

    def test_openid_redirect(self):
        response = self.fetch("/openid/client/login", follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue("/openid/server/authenticate?" in response.headers["Location"])

    def test_openid_get_user(self):
        response = self.fetch(
            "/openid/client/login?openid.mode=blah"
            "&openid.ns.ax=http://openid.net/srv/ax/1.0"
            "&openid.ax.type.email=http://axschema.org/contact/email"
            "&openid.ax.value.email=foo@example.com"
        )
        response.rethrow()
        parsed = json_decode(response.body)
        self.assertEqual(parsed["email"], "foo@example.com")

    def test_oauth10_redirect(self):
        response = self.fetch("/oauth10/client/login", follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue(
            response.headers["Location"].endswith(
                "/oauth1/server/authorize?oauth_token=zxcv"
            )
        )
        # the cookie is base64('zxcv')|base64('1234')
        self.assertTrue(
            '_oauth_request_token="enhjdg==|MTIzNA=="'
            in response.headers["Set-Cookie"],
            response.headers["Set-Cookie"],
        )

    def test_oauth10_get_user(self):
        response = self.fetch(
            "/oauth10/client/login?oauth_token=zxcv",
            headers={"Cookie": "_oauth_request_token=enhjdg==|MTIzNA=="},
        )
        response.rethrow()
        parsed = json_decode(response.body)
        self.assertEqual(parsed["email"], "foo@example.com")
        self.assertEqual(parsed["access_token"], dict(key="uiop", secret="5678"))

    def test_oauth10_request_parameters(self):
        response = self.fetch("/oauth10/client/request_params")
        response.rethrow()
        parsed = json_decode(response.body)
        self.assertEqual(parsed["oauth_consumer_key"], "asdf")
        self.assertEqual(parsed["oauth_token"], "uiop")
        self.assertTrue("oauth_nonce" in parsed)
        self.assertTrue("oauth_signature" in parsed)

    def test_oauth10a_redirect(self):
        response = self.fetch("/oauth10a/client/login", follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue(
            response.headers["Location"].endswith(
                "/oauth1/server/authorize?oauth_token=zxcv"
            )
        )
        # the cookie is base64('zxcv')|base64('1234')
        self.assertTrue(
            '_oauth_request_token="enhjdg==|MTIzNA=="'
            in response.headers["Set-Cookie"],
            response.headers["Set-Cookie"],
        )

    @unittest.skipIf(mock is None, "mock package not present")
    def test_oauth10a_redirect_error(self):
        with mock.patch.object(OAuth1ServerRequestTokenHandler, "get") as get:
            get.side_effect = Exception("boom")
            with ExpectLog(app_log, "Uncaught exception"):
                response = self.fetch("/oauth10a/client/login", follow_redirects=False)
            self.assertEqual(response.code, 500)

    def test_oauth10a_get_user(self):
        response = self.fetch(
            "/oauth10a/client/login?oauth_token=zxcv",
            headers={"Cookie": "_oauth_request_token=enhjdg==|MTIzNA=="},
        )
        response.rethrow()
        parsed = json_decode(response.body)
        self.assertEqual(parsed["email"], "foo@example.com")
        self.assertEqual(parsed["access_token"], dict(key="uiop", secret="5678"))

    def test_oauth10a_request_parameters(self):
        response = self.fetch("/oauth10a/client/request_params")
        response.rethrow()
        parsed = json_decode(response.body)
        self.assertEqual(parsed["oauth_consumer_key"], "asdf")
        self.assertEqual(parsed["oauth_token"], "uiop")
        self.assertTrue("oauth_nonce" in parsed)
        self.assertTrue("oauth_signature" in parsed)

    def test_oauth10a_get_user_coroutine_exception(self):
        response = self.fetch(
            "/oauth10a/client/login_coroutine?oauth_token=zxcv&fail_in_get_user=true",
            headers={"Cookie": "_oauth_request_token=enhjdg==|MTIzNA=="},
        )
        self.assertEqual(response.code, 503)

    def test_oauth2_redirect(self):
        response = self.fetch("/oauth2/client/login", follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue("/oauth2/server/authorize?" in response.headers["Location"])

    def test_facebook_login(self):
        response = self.fetch("/facebook/client/login", follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue("/facebook/server/authorize?" in response.headers["Location"])
        response = self.fetch(
            "/facebook/client/login?code=1234", follow_redirects=False
        )
        self.assertEqual(response.code, 200)
        user = json_decode(response.body)
        self.assertEqual(user["access_token"], "asdf")
        self.assertEqual(user["session_expires"], "3600")

    def base_twitter_redirect(self, url):
        # Same as test_oauth10a_redirect
        response = self.fetch(url, follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue(
            response.headers["Location"].endswith(
                "/oauth1/server/authorize?oauth_token=zxcv"
            )
        )
        # the cookie is base64('zxcv')|base64('1234')
        self.assertTrue(
            '_oauth_request_token="enhjdg==|MTIzNA=="'
            in response.headers["Set-Cookie"],
            response.headers["Set-Cookie"],
        )

    def test_twitter_redirect(self):
        self.base_twitter_redirect("/twitter/client/login")

    def test_twitter_redirect_gen_coroutine(self):
        self.base_twitter_redirect("/twitter/client/login_gen_coroutine")

    def test_twitter_authenticate_redirect(self):
        response = self.fetch("/twitter/client/authenticate", follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue(
            response.headers["Location"].endswith(
                "/twitter/server/authenticate?oauth_token=zxcv"
            ),
            response.headers["Location"],
        )
        # the cookie is base64('zxcv')|base64('1234')
        self.assertTrue(
            '_oauth_request_token="enhjdg==|MTIzNA=="'
            in response.headers["Set-Cookie"],
            response.headers["Set-Cookie"],
        )

    def test_twitter_get_user(self):
        response = self.fetch(
            "/twitter/client/login?oauth_token=zxcv",
            headers={"Cookie": "_oauth_request_token=enhjdg==|MTIzNA=="},
        )
        response.rethrow()
        parsed = json_decode(response.body)
        self.assertEqual(
            parsed,
            {
                "access_token": {
                    "key": "hjkl",
                    "screen_name": "foo",
                    "secret": "vbnm",
                },
                "name": "Foo",
                "screen_name": "foo",
                "username": "foo",
            },
        )

    def test_twitter_show_user(self):
        response = self.fetch("/twitter/client/show_user?name=somebody")
        response.rethrow()
        self.assertEqual(
            json_decode(response.body), {"name": "Somebody", "screen_name": "somebody"}
        )

    def test_twitter_show_user_error(self):
        response = self.fetch("/twitter/client/show_user?name=error")
        self.assertEqual(response.code, 500)
        self.assertEqual(response.body, b"error from twitter request")


class GoogleLoginHandler(RequestHandler, GoogleOAuth2Mixin):
    def initialize(self, test):
        self.test = test
        self._OAUTH_REDIRECT_URI = test.get_url("/client/login")
        self._OAUTH_AUTHORIZE_URL = test.get_url("/google/oauth2/authorize")
        self._OAUTH_ACCESS_TOKEN_URL = test.get_url("/google/oauth2/token")

    @gen.coroutine
    def get(self):
        code = self.get_argument("code", None)
        if code is not None:
            # retrieve authenticate google user
            access = yield self.get_authenticated_user(self._OAUTH_REDIRECT_URI, code)
            user = yield self.oauth2_request(
                self.test.get_url("/google/oauth2/userinfo"),
                access_token=access["access_token"],
            )
            # return the user and access token as json
            user["access_token"] = access["access_token"]
            self.write(user)
        else:
            self.authorize_redirect(
                redirect_uri=self._OAUTH_REDIRECT_URI,
                client_id=self.settings["google_oauth"]["key"],
                scope=["profile", "email"],
                response_type="code",
                extra_params={"prompt": "select_account"},
            )


class GoogleOAuth2AuthorizeHandler(RequestHandler):
    def get(self):
        # issue a fake auth code and redirect to redirect_uri
        code = "fake-authorization-code"
        self.redirect(url_concat(self.get_argument("redirect_uri"), dict(code=code)))


class GoogleOAuth2TokenHandler(RequestHandler):
    def post(self):
        assert self.get_argument("code") == "fake-authorization-code"
        # issue a fake token
        self.finish(
            {"access_token": "fake-access-token", "expires_in": "never-expires"}
        )


class GoogleOAuth2UserinfoHandler(RequestHandler):
    def get(self):
        assert self.get_argument("access_token") == "fake-access-token"
        # return a fake user
        self.finish({"name": "Foo", "email": "foo@example.com"})


class GoogleOAuth2Test(AsyncHTTPTestCase):
    def get_app(self):
        return Application(
            [
                # test endpoints
                ("/client/login", GoogleLoginHandler, dict(test=self)),
                # simulated google authorization server endpoints
                ("/google/oauth2/authorize", GoogleOAuth2AuthorizeHandler),
                ("/google/oauth2/token", GoogleOAuth2TokenHandler),
                ("/google/oauth2/userinfo", GoogleOAuth2UserinfoHandler),
            ],
            google_oauth={
                "key": "fake_google_client_id",
                "secret": "fake_google_client_secret",
            },
        )

    def test_google_login(self):
        response = self.fetch("/client/login")
        self.assertDictEqual(
            {
                "name": "Foo",
                "email": "foo@example.com",
                "access_token": "fake-access-token",
            },
            json_decode(response.body),
        )
