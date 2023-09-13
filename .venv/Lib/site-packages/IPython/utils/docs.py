import os

GENERATING_DOCUMENTATION = os.environ.get("IN_SPHINX_RUN", None) == "True"
