import numpy as np

from route_reconstruction.reconstruction import find_target_blocks


def test_find_target_blocks():
    mask = np.array([False, True, True, False, True, False, True, True, True])
    blocks = find_target_blocks(mask)
    assert blocks == [(1, 2), (4, 4), (6, 8)]
