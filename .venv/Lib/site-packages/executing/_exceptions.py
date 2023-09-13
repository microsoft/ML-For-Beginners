
class KnownIssue(Exception):
    """
    Raised in case of an known problem. Mostly because of cpython bugs.
    Executing.node gets set to None in this case.
    """

    pass


class VerifierFailure(Exception):
    """
    Thrown for an unexpected mapping from instruction to ast node
    Executing.node gets set to None in this case.
    """

    def __init__(self, title, node, instruction):
        # type: (object, object, object) -> None
        self.node = node
        self.instruction = instruction

        super().__init__(title) # type: ignore[call-arg]
