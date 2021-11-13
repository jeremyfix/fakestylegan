#!/bin/env python3

# Copyright (c) 2021, Jeremy Fix. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


# Standard imports
import sys
# External imports
import numpy as np
import PIL.Image
# Local imports
import align

aligner = align.Aligner()

# Load in-the-wild image.
if len(sys.argv) != 2:
    print(f"Usage: {' '.join(sys.argv)} image")
    sys.exit(-1)

img = PIL.Image.open(sys.argv[1])
img = aligner(img)
img.show()
