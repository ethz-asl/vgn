import enum


class Outcome(enum.Enum):
    """Possible outcomes of a grasp experiment."""

    COLLISION = 1
    EMPTY = 2
    SLIPPED = 3
    SUCCESS = 4
    ROBUST = 5
