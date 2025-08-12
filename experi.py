import re
match = re.search(r"\b\d+", "1st on")  # first integer only

if match:
    number = int(match.group())
    print(number)  # 42