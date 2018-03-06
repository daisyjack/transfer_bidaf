def get_cur_path1():
    import os
    # return os.path.abspath(os.curdir)
    return __file__
    # return os.path.dirname(__file__)

import sys
print(sys.path)