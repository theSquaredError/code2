import numpy as np
max_DIMENSIONALITY = 10
min_DIMENSIONALITY = -10
ACTION_GUIDE = 2
ACTION_UTTER = 0
ACTION_POINT = 1
ACTION_SPACE = np.arange(3)
n_octants = 8
n_segments = 10
n_blocks = n_octants+n_segments
n_vertices = 10