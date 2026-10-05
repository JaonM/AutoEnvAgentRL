"""Append-only metric records; checkpoints commit a verified byte prefix."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid


class MetricJournal:
    def __init__(self, output, name, *, cursor=None, legacy=()):
        self.path = Path(output) / f'{name}.jsonl'
        self.name = name
        self.count = 0
        self.last = None
        self.digest = hashlib.sha256()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        limit = cursor['offset'] if cursor else 0
        with self.path.open('r+b') as stream:
            if self.path.stat().st_size < limit:
                raise ValueError(f'truncated committed metric journal: {name}')
            while stream.tell() < limit:
                line = stream.readline()
                if stream.tell() > limit or not line.endswith(b'\n'):
                    raise ValueError(f'invalid metric journal cursor: {name}')
                self.last = json.loads(line)
                self.digest.update(line)
                self.count += 1
            if cursor and (self.digest.hexdigest() != cursor['sha256'] or self.count != cursor['count']):
                raise ValueError(f'committed metric journal checksum mismatch: {name}')
            if stream.read(1):
                stream.seek(limit)
                recovery = self.path.parent / 'recovery'
                recovery.mkdir(exist_ok=True)
                with (recovery / f'{name}-{uuid.uuid4().hex}.jsonl').open('wb') as tail:
                    shutil.copyfileobj(stream, tail)
                    tail.flush()
                    os.fsync(tail.fileno())
                stream.truncate(limit)
                stream.flush()
                os.fsync(stream.fileno())
        self.stream = self.path.open('ab', buffering=0)
        if not cursor:
            for record in legacy:
                self.append(record)

    def append(self, record):
        line = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode()
        remaining = memoryview(line)
        while remaining:
            written = self.stream.write(remaining)
            if not written:
                raise OSError('metric journal write made no progress')
            remaining = remaining[written:]
        self.digest.update(line)
        self.last = json.loads(line)
        self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if index != -1 or not self.count:
            raise IndexError(index)
        return self.last

    def __iter__(self):
        with self.path.open('rb') as stream:
            for _ in range(self.count):
                yield json.loads(stream.readline())

    def cursor(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return {'offset': self.stream.tell(), 'count': self.count, 'sha256': self.digest.hexdigest()}

    def export(self):
        target = self.path.with_suffix('.json')
        temporary = target.with_name('.' + target.name + '.tmp')
        with temporary.open('w') as stream:
            stream.write('[\n')
            for index, row in enumerate(self):
                if index:
                    stream.write(',\n')
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            stream.write('\n]\n')
        temporary.replace(target)

    def close(self):
        self.stream.close()
