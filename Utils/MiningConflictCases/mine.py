
with open("./conflict.tsv", "r", encoding="utf-8") as f:
    with open("./en-us.tsv", "w", encoding="utf-8") as w:
        for line in f:
            row = line.strip().split("\t")
            if(row[4].lower() == "en-us"):
                w.write(line)