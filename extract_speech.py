from utils import *
import re


def extract_speech_string(hslld_string):
    output = ''

    for line in hslld_string.split('\n'):
        # check that this line contains speaking
        if not line.startswith('*'):
            continue

        # delete beginning of line
        line = line[6:]

        # remove parenthesis and angle brackets
        line = re.sub(r'[\(\)\<\>]', '', line)

        # remove [brackets and their contents]
        line = re.sub(r'\[.*?\]', '', line)

        # remove +/
        line = re.sub(r'\+\/', '', line)

        # make &=noises into *noises*
        line = re.sub(r'&=?(\w+)', r'*\1*', line)

        # remove @ and following letter
        line = re.sub(r'@.', '', line)

        # replaces underscores
        line = re.sub(r'_', ' ', line)

        # remove trailing spaces
        line = re.sub(r' +([\?\!\.])', r'\1', line)

        # fix spaces
        line = re.sub(r' +', ' ', line)

        # finished cleaning

        # if the line consists of 1 character, we don't need it
        if re.match(r'^(.)\1{0,}[\?\!\.]$', line):
            continue

        output += line + '\n'

    return output


if __name__ == "__main__":
    file_path = data_path + 'HV1/MT/admmt1.cha'

    with open(file_path, 'r') as f:
        content = f.read()

    clean_file = extract_speech_string(content)

    print(clean_file)
