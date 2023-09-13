from pandas.core.internals.api import make_block
from pandas.core.internals.array_manager import (
    ArrayManager,
    SingleArrayManager,
)
from pandas.core.internals.base import (
    DataManager,
    SingleDataManager,
)
from pandas.core.internals.blocks import (  # io.pytables, io.packers
    Block,
    DatetimeTZBlock,
    ExtensionBlock,
)
from pandas.core.internals.concat import concatenate_managers
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
    create_block_manager_from_blocks,
)

__all__ = [
    "Block",
    "DatetimeTZBlock",
    "ExtensionBlock",
    "make_block",
    "DataManager",
    "ArrayManager",
    "BlockManager",
    "SingleDataManager",
    "SingleBlockManager",
    "SingleArrayManager",
    "concatenate_managers",
    # this is preserved here for downstream compatibility (GH-33892)
    "create_block_manager_from_blocks",
]


def __getattr__(name: str):
    import warnings

    from pandas.util._exceptions import find_stack_level

    if name in ["NumericBlock", "ObjectBlock"]:
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )
        if name == "NumericBlock":
            from pandas.core.internals.blocks import NumericBlock

            return NumericBlock
        else:
            from pandas.core.internals.blocks import ObjectBlock

            return ObjectBlock

    raise AttributeError(f"module 'pandas.core.internals' has no attribute '{name}'")
