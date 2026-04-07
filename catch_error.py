import traceback
import sys

try:
    import app
except Exception as e:
    with open("error_log.txt", "w") as f:
        traceback.print_exc(file=f)
