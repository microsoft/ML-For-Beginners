"""Manager to read and modify config data in JSON files.
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import errno
import json
import os
from typing import Any

from traitlets.config import LoggingConfigurable
from traitlets.traitlets import Unicode


def recursive_update(target: dict[Any, Any], new: dict[Any, Any]) -> None:
    """Recursively update one dictionary using another.

    None values will delete their keys.
    """
    for k, v in new.items():
        if isinstance(v, dict):
            if k not in target:
                target[k] = {}
            recursive_update(target[k], v)
            if not target[k]:
                # Prune empty subdicts
                del target[k]

        elif v is None:
            target.pop(k, None)

        else:
            target[k] = v


class BaseJSONConfigManager(LoggingConfigurable):
    """General JSON config manager

    Deals with persisting/storing config in a json file
    """

    config_dir = Unicode(".")

    def ensure_config_dir_exists(self) -> None:
        try:
            os.makedirs(self.config_dir, 0o755)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def file_name(self, section_name: str) -> str:
        return os.path.join(self.config_dir, section_name + ".json")

    def get(self, section_name: str) -> Any:
        """Retrieve the config data for the specified section.

        Returns the data as a dictionary, or an empty dictionary if the file
        doesn't exist.
        """
        filename = self.file_name(section_name)
        if os.path.isfile(filename):
            with open(filename, encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def set(self, section_name: str, data: Any) -> None:
        """Store the given config data."""
        filename = self.file_name(section_name)
        self.ensure_config_dir_exists()

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def update(self, section_name: str, new_data: Any) -> Any:
        """Modify the config section by recursively updating it with new_data.

        Returns the modified config data as a dictionary.
        """
        data = self.get(section_name)
        recursive_update(data, new_data)
        self.set(section_name, data)
        return data
