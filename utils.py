import glob


data_path = '../Original Transcripts/'

data_folders = [ f'HV{i}/MT/' for i in [1, 2, 3, 5, 7] ]


def get_all_transcript_paths():
    return glob.glob(data_path + 'HV*/MT/*.cha')


def get_all_transcript_strings():
    return [safe_read(f) for f in get_all_transcript_paths()]


def safe_read(file):
    with open(file, 'rb') as f:
        meat = f.read().decode()
    return meat


if __name__ == "__main__":
    # test loading
    from time import time

    start_time = time()
    all_strings = get_all_transcript_strings()
    end_time = time()

    print(f'loaded {len(all_strings)} files in {end_time - start_time} seconds')
