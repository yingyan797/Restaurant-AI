import re

i = 2
while i > 1:
    try:
        i += 1
        if i > 293850297:
            i = 2
    except KeyboardInterrupt as k:
        print("Don't interrupt me, i=", i)
        
