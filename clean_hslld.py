'''
extract the speech from the whole dataset, and save each file under the same folder as the
'''
from utils import get_all_transcript_paths, safe_read
from extract_speech import extract_speech_string
from tqdm import tqdm


if __name__ == '__main__':
    for path in get_all_transcript_paths():
        transcript = extract_speech_string(safe_read(path))

        with open(f'transcripts/{path.parts[-1]}', 'w') as f:
            f.write(transcript)
