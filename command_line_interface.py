# Copyright (C) 2023 HydrusBeta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import controllable_talknet

import librosa
import soundfile

import sys
import base64
import io
import os


def generate_audio(user_text, input_path, character, pitch_factor, pitch_options) -> (float, int):
    _, f0s, f0s_wo_silence, wav_name = controllable_talknet.select_file.__wrapped__(input_path, [''])
    src, _, _, _ = controllable_talknet.generate_audio.__wrapped__(0, character, None, user_text, pitch_options,
                                                                   pitch_factor, wav_name, f0s, f0s_wo_silence)
    return get_audio_from_src(src, encoding='ascii')


def get_audio_from_src(src, encoding):
    _, raw = src.split(',')
    b64_output_bytes = raw.encode(encoding)
    output_bytes = base64.b64decode(b64_output_bytes)
    buffer = io.BytesIO(output_bytes)
    return librosa.load(buffer, sr=None)


if __name__ == '__main__':
    # parse arguments
    user_text = sys.argv[1]
    input_path = sys.argv[2]
    character = sys.argv[3]
    pitch_factor = sys.argv[4]
    pitch_options = sys.argv[5:]

    # generate audio
    output_array, output_samplerate = generate_audio(user_text, input_path, character, pitch_factor, pitch_options)

    # prepare output directory
    results_dir = os.path.join(controllable_talknet.RUN_PATH, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # write output file
    input_filename = os.path.basename(input_path)
    input_filename_sans_extension = input_filename.split('.')[0]
    output_filename = os.path.join(results_dir, input_filename_sans_extension + '.flac')
    soundfile.write(output_filename, output_array, output_samplerate, format='FLAC')
