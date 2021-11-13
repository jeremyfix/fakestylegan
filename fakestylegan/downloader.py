# Copyright (c) 2021, Jeremy Fix. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


# Standard imports
import sys
import urllib.request
from pathlib import Path
import bz2

DEFAULT_READ_SIZE = 1024

def downloadfile(url: str,
                 filename: Path,
                 isbz2=False):
    if isbz2:
        decompressor = bz2.BZ2Decompressor()

    with urllib.request.urlopen(url) as response:
        totsize = int(response.getheader('Content-Length'))
        size = 0
        with open(filename, 'wb') as outfile:
            while size != totsize:
                data = response.read(DEFAULT_READ_SIZE)
                size += len(data)
                if isbz2:
                    data = decompressor.decompress(data)
                outfile.write(data)
                sys.stdout.write('\r ' + f"{size/1024**2:.2f} Mo / {totsize/1024**2:.2f} Mo")
                sys.stdout.flush()
        sys.stdout.write('\n')
