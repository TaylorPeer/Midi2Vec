import os
import logging

from midi_to_dataframe import NoteMapper, MidiReader, MidiWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_directory(midi_directory, note_mapping_config_path, output_dir):
    """
    Processes all MIDI files found in a given directory (recursively) and TODO.
    :param midi_directory: the directory to process.
    :param note_mapping_config_path: path to MIDI to text configuration settings.
    :param output_dir: TODO
    :return: none
    """

    note_mapper = NoteMapper(note_mapping_config_path)

    for subdir, dirs, files in os.walk(midi_directory):

        for f in files:
            if f.lower().endswith(".mid"):

                path = os.path.join(subdir, f)
                print(f)

                try:
                    midi_reader = MidiReader(note_mapper)
                    midi_writer = MidiWriter(note_mapper)
                    df = midi_reader.convert_to_dataframe(path)

                    for index, row in df.iterrows():
                        notes = row['notes'].split(",")
                        fixed_notes = []
                        for note in notes:
                            if "_" in note:
                                fields = note.split("_")
                                instrument = "bass"
                                pitch = fields[1]
                                duration = float(fields[2])
                                fixed_note = instrument + "_" + pitch + "_" + str(duration)
                                fixed_notes.append(fixed_note)
                        notes = ','.join(fixed_notes)
                        if len(fixed_notes) > 0:
                            df.at[index, 'notes'] = notes

                    midi_writer.convert_to_midi(df, output_dir + "/" + f)

                except Exception:
                    logger.error("Error creating sequence for file: " + str(f), exc_info=True)


def main():
    output_dir = '/Users/taylorpeer/Projects/se-project/Midi2Vec/resources/midi/bassline/UKBasslineMidiFiles_SoulRush/Basslines/fixed'

    corpus_dir = "/Users/taylorpeer/Projects/se-project/Midi2Vec/resources/midi/bassline/UKBasslineMidiFiles_SoulRush/Basslines/midi"
    note_mapping_config_path = "../settings/map-to-group.json"
    process_directory(corpus_dir, note_mapping_config_path, output_dir)


if __name__ == '__main__':
    main()
