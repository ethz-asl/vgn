from __future__ import division


def workspace_lines(size):
    return [
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, size, 0.0],
        [size, size, 0.0],
        [0.0, size, 0.0],
        [0.0, size, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, size],
        [size, 0.0, size],
        [size, size, size],
        [size, size, size],
        [0.0, size, size],
        [0.0, size, size],
        [0.0, 0.0, size],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, 0.0],
        [size, 0.0, size],
        [size, size, 0.0],
        [size, size, size],
        [0.0, size, 0.0],
        [0.0, size, size],
    ]


def gripper_lines(width, depth):
    return [
        [0.0, 0.0, -depth / 2.0],
        [0.0, 0.0, 0.0],
        [0.0, -width / 2.0, 0.0],
        [0.0, -width / 2.0, depth],
        [0.0, width / 2.0, 0.0],
        [0.0, width / 2.0, depth],
        [0.0, -width / 2.0, 0.0],
        [0.0, width / 2.0, 0.0],
    ]
