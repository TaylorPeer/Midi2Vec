import os
import logging

from midi_to_dataframe import NoteMapper, MidiReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_directory(midi_directory, note_mapping_config_path, output_file_name):
    """
    Processes all MIDI files found in a given directory (recursively) and converts them into text representations for
    training semantic models.
    :param midi_directory: the directory to process.
    :param note_mapping_config_path: path to MIDI to text configuration settings.
    :param output_file_name: full file path and name to output file.
    :return: none
    """

    note_mapper = NoteMapper(note_mapping_config_path)
    output_file = open(output_file_name, "w")

    for subdir, dirs, files in os.walk(midi_directory):

        for f in files:
            if f.lower().endswith(".mid"):

                path = os.path.join(subdir, f)
                print(f)

                try:
                    midi_reader = MidiReader(note_mapper)
                    df = midi_reader.convert_to_dataframe(path)

                    grouped = df.groupby('measure')['notes'].apply(','.join)
                    for measure, notes in grouped.items():
                        output_file.write(notes + "\n")

                except Exception:
                    logger.error("Error creating sequence for file: " + str(f), exc_info=True)

    output_file.close()


def main():
    # Output file
    file_path = '/Users/taylorpeer/Projects/se-project/midi-embeddings/data'
    file_name = "full_1_measure.line"
    file_full = os.path.join(file_path, file_name)

    corpus_dir = "/Users/taylorpeer/Projects/se-project/midi-embeddings/data/corpora/web-midi"
    note_mapping_config_path = "../settings/map-to-group.json"
    process_directory(corpus_dir, note_mapping_config_path, file_full)


if __name__ == '__main__':
    main()
