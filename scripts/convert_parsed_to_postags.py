import os


def read_file(fname):
    examples = []
    one_sample = []
    with open(fname, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                tok_line = line.split()
                if tok_line[0] == '1':
                    examples.append(one_sample)
                    one_sample = []
                one_sample.append(tok_line)
        examples.append(one_sample)
    return examples


def convert_to_pos(examples):
    new_examples = []
    for sample in examples:
        pos = []
        for sentence in sample:
            pos.append(sentence[3])
        new_examples.append(pos)
    return new_examples


def save_new_file(fname, pos_examples):
    print('Saving %s...' % fname)
    f = open(fname, 'w', encoding='utf8')
    for pos in pos_examples:
        s = ' '.join(pos)
        if s:
            f.write(s + '\n')
    f.close()


if __name__ == '__main__':
    for dname in os.listdir('.'):
        if not os.path.isdir(dname):
            continue
        prefix = 'train'
        if 'test' in dname:
            # prefix = 'test.2017'
            prefix = 'test'
        elif 'dev' in dname:
            prefix = 'dev'
        if ('training' in dname and 'dev' in dname) or ('.py' in dname):
            continue
        try:
            source_parsed = os.path.join(dname, prefix + '.src.parsed')
            target_parsed = os.path.join(dname, prefix + '.mt.parsed')
            examples_source = read_file(source_parsed)
            examples_target = read_file(target_parsed)
            pos_source = convert_to_pos(examples_source)
            pos_target = convert_to_pos(examples_target)
            save_new_file(os.path.join(dname, prefix + '.src.pos'), pos_source)
            save_new_file(os.path.join(dname, prefix + '.mt.pos'), pos_target)
        except FileNotFoundError as e:
            print('Skipping dir; parsed file not found: {}'.format(e))
            continue
