#!/usr/bin/env python3
"""Test real WAV loading without a model: python3 tests/test-read-wav.py.

Requires a C++11 compiler (defaults to c++; override with CXX).
Builds the unchanged common.cpp and bundled dr_wav decoder in a temporary directory.
"""

import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = r'''
#include "common.h"
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iterator>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    std::string input = argv[2];
    if (std::string(argv[1]) == "buffer") {
        std::ifstream file(input, std::ios::binary);
        input.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    std::vector<float> mono;
    std::vector<std::vector<float>> stereo;
    if (!read_wav(input, mono, stereo, std::string(argv[3]) == "stereo")) return 1;
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    const uint64_t sizes[] = {mono.size(), stereo.size(), stereo.empty() ? 0 : stereo[0].size()};
    std::fwrite(sizes, sizeof(uint64_t), 3, stdout);
    std::fwrite(mono.data(), sizeof(float), mono.size(), stdout);
    for (const auto & channel : stereo) {
        std::fwrite(channel.data(), sizeof(float), channel.size(), stdout);
    }
}
'''


def chunk(name, data):
    return name + struct.pack('<I', len(data)) + data + b'\0' * (len(data) % 2)


def wav(channels, samples, metadata_size, metadata_after):
    fmt = struct.pack('<HHIIHH', 1, channels, 16000, 32000 * channels, 2 * channels, 16)
    metadata = chunk(b'JUNK', b'x' * metadata_size) if metadata_size else b''
    pcm = chunk(b'data', struct.pack('<' + 'h' * len(samples), *samples))
    body = b'WAVE' + chunk(b'fmt ', fmt)
    body += (pcm + metadata) if metadata_after else (metadata + pcm)
    return b'RIFF' + struct.pack('<I', len(body)) + body


class ReadWavTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='whisper-read-wav-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        source = cls.directory / 'runner.cpp'
        source.write_text(RUNNER)
        cls.runner = cls.directory / ('runner.exe' if os.name == 'nt' else 'runner')
        command = shlex.split(os.environ.get('CXX', 'c++'))
        command += shlex.split(os.environ.get('CXXFLAGS', ''))
        command += ['-std=c++11', '-I', str(ROOT / 'examples'), str(source),
                    str(ROOT / 'examples/common.cpp'), '-o', str(cls.runner)]
        subprocess.run(command, check=True)

    def check_fixture(self, channels, metadata_size, metadata_after, diarize=False,
                      unknown_length=False):
        # Exact binary fractions include both signed extrema, silence and ordinary samples.
        samples = [-32768, -16384, -1, 0, 1, 8192, 16384, 32767] * channels
        expected = [sum(samples[i:i + channels]) / (32768.0 * channels)
                    for i in range(0, len(samples), channels)]
        output_channels = [[value / 32768.0 for value in samples[channel::channels]]
                           for channel in range(channels)] if diarize else []
        audio = wav(channels, samples, metadata_size, metadata_after)
        if unknown_length:
            # FFmpeg cannot seek back when writing to a pipe, so both sizes are unknown.
            audio = audio[:4] + b'\xff' * 4 + audio[8:]
            data_offset = audio.index(b'data') + 4
            audio = audio[:data_offset] + b'\xff' * 4 + audio[data_offset + 4:]
        path = self.directory / 'fixture.wav'
        path.write_bytes(audio)
        for mode in (('stdin',) if unknown_length else ('file', 'buffer', 'stdin')):
            with self.subTest(mode=mode):
                result = subprocess.run([str(self.runner), mode,
                                         '-' if mode == 'stdin' else str(path),
                                         'stereo' if diarize else 'mono'],
                                        input=audio if mode == 'stdin' else None,
                                        capture_output=True, check=True)
                count, channel_count, channel_length = struct.unpack('=QQQ', result.stdout[:24])
                self.assertEqual(count, len(expected))
                self.assertEqual(channel_count, len(output_channels))
                self.assertEqual(channel_length, len(expected) if diarize else 0)
                values = expected + [value for channel in output_channels for value in channel]
                self.assertEqual(len(result.stdout), 24 + len(values) * 4)
                self.assertEqual(struct.unpack('=' + 'f' * len(values), result.stdout[24:]),
                                 tuple(values))

    def test_unknown_length_mono_pipe(self):
        self.check_fixture(1, 0, False, unknown_length=True)

    def test_unknown_length_stereo_pipe(self):
        self.check_fixture(2, 0, False, unknown_length=True)

    def test_unknown_length_diarization_pipe(self):
        self.check_fixture(2, 0, False, diarize=True, unknown_length=True)


def make_test(channels, metadata_size, metadata_after, diarize):
    def test(self):
        self.check_fixture(channels, metadata_size, metadata_after, diarize)
    return test


for channels, diarize in ((1, False), (2, False), (2, True)):
    for size, after in ((0, False), (1, False), (1, True), (64000, False), (64000, True)):
        name = 'test_%d_channels_%d_metadata_%s_%s' % (
            channels, size, 'after' if after else 'before', 'diarize' if diarize else 'mono')
        setattr(ReadWavTests, name, make_test(channels, size, after, diarize))


if __name__ == '__main__':
    unittest.main()
