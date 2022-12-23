


def read_tsv(file_path):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(line.strip().split("\t"))
    return data
