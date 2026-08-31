import glob

for fpath in sorted(glob.glob("*.tex")):
    with open(fpath) as f:
        lines = f.readlines()
    for idx, line in enumerate(lines, 1):
        for i, char in enumerate(line):
            if char == '%':
                if i == 0 or line[i-1] != '\\':
                    prefix = line[:i]
                    if prefix.strip() != "":
                        print(f"{fpath}:{idx}:{i+1}: {line.strip()}")
                    break
