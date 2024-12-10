import csv
import os

from gtts import gTTS  # requires the internet
from playsound import playsound as ps

# save location for generated audio file.
AUDIOSAVE = './sounds/'


# extract the content of the csv file and return it as a dictionary.
def convert_CSV_to_dict(csv_file: str) -> dict:
    rows = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return (rows)


# extract the content of the dictionary and use TTS to make audio files.
def convert_CSV_to_tts(csv_content: dict) -> None:
    language = 'en'
    domain = 'co.uk'

    for row in csv_content:
        tts = gTTS(text=row["yes_text"], lang=language, tld=domain)
        tts.save(f'{AUDIOSAVE}scene-{row["scene"]}-yes.mp3')
        tts = gTTS(text=row["no_text"], lang=language, tld=domain)
        tts.save(f'{AUDIOSAVE}scene-{row["scene"]}-no.mp3')

def drama_manager(scene: int, choice: list) -> tuple:
  emotion = choice[scene]["wanted_emotion"]
  return(scene, emotion)

# play the requested utterance.
def play_utterance(utterance_file: str) -> None:
    ps(utterance_file)


def main() -> None:
    utterance_list = convert_CSV_to_dict('test_workshop3-2.csv')
    # print(utterance_list[0]["wanted_emotion"])
    # test = drama_manager(1, utterance_list)
    # print(test[1])
    convert_CSV_to_tts(utterance_list)
    # play_utterance(f'{AUDIOSAVE}scene-1-no.mp3')


if __name__ == "__main__":
    main()
