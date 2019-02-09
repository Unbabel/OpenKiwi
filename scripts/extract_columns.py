def extract_columns(filepath, columns, separator='\t'):
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            if line:
                fields = line.split(separator)
                print(separator.join([fields[i - 1] for i in columns]))
            else:
                print()


if __name__ == '__main__':
    import sys

    filepath = sys.argv[1]
    columns = [int(c) for c in sys.argv[2:]]
    extract_columns(filepath, columns)
