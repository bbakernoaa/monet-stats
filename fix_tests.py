with open("tests/test_stats_function.py", "r") as f:
    content = f.read()

content = content.replace('"CCC",', '"CCC",\n        "MNE",\n        "NMSE",')
# Avoid double adding if I already added it once
content = content.replace('"MNE",\n        "NMSE",\n        "MNE",\n        "NMSE",', '"MNE",\n        "NMSE",')

with open("tests/test_stats_function.py", "w") as f:
    f.write(content)
