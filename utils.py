from pathlib import Path


data_path = Path('../Original Transcripts/')
answer_path = Path('../Ground Truth/')

all_transcripts_glob = 'HV*/MT/*.cha'

data_folders = [ f'HV{i}/MT/' for i in [1, 2, 3, 5, 7] ]

pickle_path = 'pickles/'


def get_all_transcript_paths():
    return data_path.glob(all_transcripts_glob)


def get_all_transcript_strings():
    return [safe_read(f) for f in get_all_transcript_paths()]


def safe_read(file):
    with open(file, 'r') as f:
        meat = f.read()
    return meat


if __name__ == "__main__":
    # test loading
    from time import time

    start_time = time()
    all_strings = get_all_transcript_strings()
    end_time = time()

    print(f'loaded {len(all_strings)} files in {end_time - start_time} seconds')
