# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ONNX Model Hub

This implements the python client for the ONNX model hub.
"""
import hashlib
import json
import os
import sys
import tarfile
from io import BytesIO
from os.path import join
from typing import IO, Any, Dict, List, Optional, Set, Tuple, cast
from urllib.error import HTTPError
from urllib.request import urlopen

import onnx

if "ONNX_HOME" in os.environ:
    _ONNX_HUB_DIR = join(os.environ["ONNX_HOME"], "hub")
elif "XDG_CACHE_HOME" in os.environ:
    _ONNX_HUB_DIR = join(os.environ["XDG_CACHE_HOME"], "onnx", "hub")
else:
    _ONNX_HUB_DIR = join(os.path.expanduser("~"), ".cache", "onnx", "hub")


class ModelInfo:
    """
    A class to represent a model's property and metadata in the ONNX Hub.
    It extracts model name, path, sha, tags, etc. from the passed in raw_model_info dict.

    Attributes:
        model: The name of the model.
        model_path: The path to the model, relative to the model zoo (https://github.com/onnx/models/) repo root.
        metadata: Additional metadata of the model, such as the size of the model, IO ports, etc.
        model_sha: The SHA256 digest of the model file.
        tags: A set of tags associated with the model.
        opset: The opset version of the model.
    """

    def __init__(self, raw_model_info: Dict[str, Any]) -> None:
        """
        Parameters:
            raw_model_info: A JSON dict containing the model info.
        """
        self.model = cast(str, raw_model_info["model"])

        self.model_path = cast(str, raw_model_info["model_path"])
        self.metadata: Dict[str, Any] = cast(Dict[str, Any], raw_model_info["metadata"])
        self.model_sha: Optional[str] = None
        if "model_sha" in self.metadata:
            self.model_sha = cast(str, self.metadata["model_sha"])

        self.tags: Set[str] = set()
        if "tags" in self.metadata:
            self.tags = set(cast(List[str], self.metadata["tags"]))

        self.opset = cast(int, raw_model_info["opset_version"])
        self.raw_model_info: Dict[str, Any] = raw_model_info

    def __str__(self) -> str:
        return f"ModelInfo(model={self.model}, opset={self.opset}, path={self.model_path}, metadata={self.metadata})"

    def __repr__(self) -> str:
        return self.__str__()


def set_dir(new_dir: str) -> None:
    """
    Sets the current ONNX hub cache location

    :param new_dir: location of new model hub cache
    """
    global _ONNX_HUB_DIR  # pylint: disable=global-statement
    _ONNX_HUB_DIR = new_dir


def get_dir() -> str:
    """
    Gets the current ONNX hub cache location

    :return: The location of the ONNX hub model cache
    """
    return _ONNX_HUB_DIR


def _parse_repo_info(repo: str) -> Tuple[str, str, str]:
    """
    Gets the repo owner, name and ref from a repo specification string.
    """
    repo_owner = repo.split(":")[0].split("/")[0]
    repo_name = repo.split(":")[0].split("/")[1]
    if ":" in repo:
        repo_ref = repo.split(":")[1]
    else:
        repo_ref = "main"
    return repo_owner, repo_name, repo_ref


def _verify_repo_ref(repo: str) -> bool:
    """
    Verifies whether the given model repo can be trusted.
    A model repo can be trusted if it matches onnx/models:main.
    """
    repo_owner, repo_name, repo_ref = _parse_repo_info(repo)
    return (repo_owner == "onnx") and (repo_name == "models") and (repo_ref == "main")


def _get_base_url(repo: str, lfs: bool = False) -> str:
    """
    Gets the base github url from a repo specification string

    :param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "main"
    :param lfs: whether the url is for downloading lfs models
    :return: the base github url for downloading
    """
    repo_owner, repo_name, repo_ref = _parse_repo_info(repo)

    if lfs:
        return f"https://media.githubusercontent.com/media/{repo_owner}/{repo_name}/{repo_ref}/"
    return f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{repo_ref}/"


def _download_file(url: str, file_name: str) -> None:
    """
    Downloads the file with specifed file_name from the url

    :param url: a url of download link
    :param file_name: a specified file name for the downloaded file
    """
    chunk_size = 16384  # 1024 * 16
    with urlopen(url) as response, open(file_name, "wb") as f:
        # Loads processively with chuck_size for huge models
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def list_models(
    repo: str = "onnx/models:main",
    model: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[ModelInfo]:
    """
    Gets the list of model info consistent with a given name and tags

    :param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "main"
    :param model: The name of the model to search for. If `None`, will return all models with matching tags.
    :param tags: A list of tags to filter models by. If `None`, will return all models with matching name.
    :return: list of ModelInfo
    """
    base_url = _get_base_url(repo)
    manifest_url = base_url + "ONNX_HUB_MANIFEST.json"
    try:
        with urlopen(manifest_url) as response:
            manifest: List[ModelInfo] = [
                ModelInfo(info) for info in json.load(cast(IO[str], response))
            ]
    except HTTPError as e:
        raise AssertionError(f"Could not find manifest at {manifest_url}") from e

    # Filter by model name first.
    matching_models = (
        manifest
        if model is None
        else [m for m in manifest if m.model.lower() == model.lower()]
    )

    # Filter by tags
    if tags is None:
        return matching_models

    canonical_tags = {t.lower() for t in tags}
    matching_info_list: List[ModelInfo] = []
    for m in matching_models:
        model_tags = {t.lower() for t in m.tags}
        if len(canonical_tags.intersection(model_tags)) > 0:
            matching_info_list.append(m)
    return matching_info_list


def get_model_info(
    model: str, repo: str = "onnx/models:main", opset: Optional[int] = None
) -> ModelInfo:
    """
    Gets the model info matching the given name and opset.

    :param model: The name of the onnx model in the manifest. This field is case-sensitive
    :param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "main"
    :param opset: The opset of the model to get. The default of `None` will return the model with largest opset.
    :return: ModelInfo
    """
    matching_models = list_models(repo, model)
    if not matching_models:
        raise AssertionError(f"No models found with name {model}")

    if opset is None:
        selected_models = sorted(matching_models, key=lambda m: -m.opset)
    else:
        selected_models = [m for m in matching_models if m.opset == opset]
        if len(selected_models) == 0:
            valid_opsets = [m.opset for m in matching_models]
            raise AssertionError(
                f"{model} has no version with opset {opset}. Valid opsets: {valid_opsets}"
            )
    return selected_models[0]


def load(
    model: str,
    repo: str = "onnx/models:main",
    opset: Optional[int] = None,
    force_reload: bool = False,
    silent: bool = False,
) -> Optional[onnx.ModelProto]:
    """
    Downloads a model by name from the onnx model hub

    :param model: The name of the onnx model in the manifest. This field is case-sensitive
    :param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "main"
    :param opset: The opset of the model to download. The default of `None` automatically chooses the largest opset
    :param force_reload: Whether to force the model to re-download even if its already found in the cache
    :param silent: Whether to suppress the warning message if the repo is not trusted.
    :return: ModelProto or None
    """
    selected_model = get_model_info(model, repo, opset)
    local_model_path_arr = selected_model.model_path.split("/")
    if selected_model.model_sha is not None:
        local_model_path_arr[
            -1
        ] = f"{selected_model.model_sha}_{local_model_path_arr[-1]}"
    local_model_path = join(_ONNX_HUB_DIR, os.sep.join(local_model_path_arr))

    if force_reload or not os.path.exists(local_model_path):
        if not _verify_repo_ref(repo) and not silent:
            msg = f"The model repo specification {repo} is not trusted and may contain security vulnerabilities. Only continue if you trust this repo."

            print(msg, file=sys.stderr)
            print("Continue?[y/n]")
            if input().lower() != "y":
                return None

        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        lfs_url = _get_base_url(repo, True)
        print(f"Downloading {model} to local path {local_model_path}")
        _download_file(lfs_url + selected_model.model_path, local_model_path)
    else:
        print(f"Using cached {model} model from {local_model_path}")

    with open(local_model_path, "rb") as f:
        model_bytes = f.read()

    if selected_model.model_sha is not None:
        downloaded_sha = hashlib.sha256(model_bytes).hexdigest()
        if not downloaded_sha == selected_model.model_sha:
            raise AssertionError(
                (
                    f"The cached model {selected_model.model} has SHA256 {downloaded_sha} while checksum should be {selected_model.model_sha}."
                    + "The model in the hub may have been updated. Use force_reload to download the model from the model hub."
                )
            )

    return onnx.load(cast(IO[bytes], BytesIO(model_bytes)))


def download_model_with_test_data(
    model: str,
    repo: str = "onnx/models:main",
    opset: Optional[int] = None,
    force_reload: bool = False,
    silent: bool = False,
) -> Optional[str]:
    """
    Downloads a model along with test data by name from the onnx model hub and returns the directory to which the files have been extracted.

    :param model: The name of the onnx model in the manifest. This field is case-sensitive
    :param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "main"
    :param opset: The opset of the model to download. The default of `None` automatically chooses the largest opset
    :param force_reload: Whether to force the model to re-download even if its already found in the cache
    :param silent: Whether to suppress the warning message if the repo is not trusted.
    :return: str or None
    """
    selected_model = get_model_info(model, repo, opset)

    local_model_with_data_path_arr = selected_model.metadata[
        "model_with_data_path"
    ].split("/")

    model_with_data_sha = selected_model.metadata["model_with_data_sha"]

    if model_with_data_sha is not None:
        local_model_with_data_path_arr[
            -1
        ] = f"{model_with_data_sha}_{local_model_with_data_path_arr[-1]}"
    local_model_with_data_path = join(
        _ONNX_HUB_DIR, os.sep.join(local_model_with_data_path_arr)
    )

    if force_reload or not os.path.exists(local_model_with_data_path):
        if not _verify_repo_ref(repo) and not silent:
            msg = f"The model repo specification {repo} is not trusted and may contain security vulnerabilities. Only continue if you trust this repo."

            print(msg, file=sys.stderr)
            print("Continue?[y/n]")
            if input().lower() != "y":
                return None

        os.makedirs(os.path.dirname(local_model_with_data_path), exist_ok=True)
        lfs_url = _get_base_url(repo, True)
        print(f"Downloading {model} to local path {local_model_with_data_path}")
        _download_file(
            lfs_url + selected_model.metadata["model_with_data_path"],
            local_model_with_data_path,
        )
    else:
        print(f"Using cached {model} model from {local_model_with_data_path}")

    with open(local_model_with_data_path, "rb") as f:
        model_with_data_bytes = f.read()

    if model_with_data_sha is not None:
        downloaded_sha = hashlib.sha256(model_with_data_bytes).hexdigest()
        if not downloaded_sha == model_with_data_sha:
            raise AssertionError(
                (
                    f"The cached model {selected_model.model} has SHA256 {downloaded_sha} while checksum should be {model_with_data_sha}."
                    + "The model in the hub may have been updated. Use force_reload to download the model from the model hub."
                )
            )

    with tarfile.open(local_model_with_data_path) as model_with_data_zipped:
        # FIXME: Avoid index manipulation with magic numbers
        local_model_with_data_dir_path = local_model_with_data_path[
            0 : len(local_model_with_data_path) - 7
        ]
        model_with_data_zipped.extractall(local_model_with_data_dir_path)
    model_with_data_path = (
        local_model_with_data_dir_path
        + "/"
        + os.listdir(local_model_with_data_dir_path)[0]
    )

    return model_with_data_path


def load_composite_model(
    network_model: str,
    preprocessing_model: str,
    network_repo: str = "onnx/models:main",
    preprocessing_repo: str = "onnx/models:main",
    opset: Optional[int] = None,
    force_reload: bool = False,
    silent: bool = False,
) -> Optional[onnx.ModelProto]:
    """
    Builds a composite model including data preprocessing by downloading a network and a preprocessing model
    and combine it into a single model

    :param model: The name of the onnx model in the manifest. This field is case-sensitive
    :param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "main"
    :param opset: The opset of the model to download. The default of `None` automatically chooses the largest opset
    :param force_reload: Whether to force the model to re-download even if its already found in the cache
    :param silent: Whether to suppress the warning message if the repo is not trusted.
    :return: ModelProto or None
    """
    preprocessing = load(
        preprocessing_model, preprocessing_repo, opset, force_reload, silent
    )
    if preprocessing is None:
        raise RuntimeError(
            f"Could not load the preprocessing model: {preprocessing_model}"
        )
    network = load(network_model, network_repo, opset, force_reload, silent)
    if network is None:
        raise RuntimeError(f"Could not load the network model: {network_model}")

    all_domains: Set[str] = set()
    domains_to_version_network: Dict[str, int] = {}
    domains_to_version_preprocessing: Dict[str, int] = {}

    for opset_import_entry in network.opset_import:
        domain = (
            "ai.onnx" if opset_import_entry.domain == "" else opset_import_entry.domain
        )
        all_domains.add(domain)
        domains_to_version_network[domain] = opset_import_entry.version

    for opset_import_entry in preprocessing.opset_import:
        domain = (
            "ai.onnx" if opset_import_entry.domain == "" else opset_import_entry.domain
        )
        all_domains.add(domain)
        domains_to_version_preprocessing[domain] = opset_import_entry.version

    preprocessing_opset_version = -1
    network_opset_version = -1
    for domain in all_domains:
        if domain == "ai.onnx":
            preprocessing_opset_version = domains_to_version_preprocessing[domain]
            network_opset_version = domains_to_version_network[domain]
        elif (
            domain in domains_to_version_preprocessing
            and domain in domains_to_version_network
            and domains_to_version_preprocessing[domain]
            != domains_to_version_preprocessing[domain]
        ):
            raise ValueError(
                f"Can not merge {preprocessing_model} and {network_model} because they contain "
                f"different opset versions for domain {domain} ({domains_to_version_preprocessing[domain]}) "
                f"and {domains_to_version_network[domain]}). Only the default domain can be "
                "automatically converted to the highest version of the two."
            )
    if preprocessing_opset_version > network_opset_version:
        network = onnx.version_converter.convert_version(
            network, preprocessing_opset_version
        )
        network.ir_version = preprocessing.ir_version
        onnx.checker.check_model(network)
    elif network_opset_version > preprocessing_opset_version:
        preprocessing = onnx.version_converter.convert_version(
            preprocessing, network_opset_version
        )
        preprocessing.ir_version = network.ir_version
        onnx.checker.check_model(preprocessing)

    io_map = []
    for out_entry, in_entry in zip(preprocessing.graph.output, network.graph.input):
        io_map.append((out_entry.name, in_entry.name))

    model_with_preprocessing = onnx.compose.merge_models(
        preprocessing, network, io_map=io_map
    )
    return model_with_preprocessing
