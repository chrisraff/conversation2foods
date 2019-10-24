# Turn files into versions that have food answers at the end
from utils import data_path, answer_path, all_transcripts_glob
from extract_speech import extract_speech_string
from pathlib import Path
import re


if __name__ == "__main__":
    all_answers = answer_path.glob(all_transcripts_glob)
    
    for answer_path in all_answers:
        assert answer_path.parts[1] == 'Ground Truth', print('The path must be wrong')

        # get the answer string
        with open(answer_path, 'r') as f:
            answer = f.read()

        # make the answer string only contain food names by removing the leading UNIQUE
        # and by removing the numbers at the start of each line
        clean_answer = ''
        for line in answer.split('\n'):
            if line == 'UNIQUE':
                continue
            if line == '':
                continue

            line = re.sub(r'[0-9]+: ?', '', line)

            clean_answer += line + '\n'

        parts = list(answer_path.parts)
        parts[1] = 'Original Transcripts'

        with open('/'.join(parts), 'r') as f:
            transcript = f.read()
        
        transcript = extract_speech_string(transcript)

        # construct an answered prompt
        prompt = "PROMPT:\n" + transcript + "\nLISTFOODS:\n" + clean_answer

        # save the answered prompt for training
        with open('training_prompts/' + parts[-1], 'w') as f:
            f.write(prompt)
