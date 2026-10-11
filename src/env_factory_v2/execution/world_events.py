"""Pure external-event interpreter shared by world execution and projection."""
import copy

from env_factory_v2.contracts.contract import require, validate_data
from env_factory_v2.contracts.expressions import canonical


def ordered_events(source):
    return sorted(enumerate(source.get('events', [])),
                  key=lambda pair: (pair[1]['at_seconds'], pair[0]))


def apply_events(source, before, events):
    candidate = copy.deepcopy(before)
    tables = {table['id']: table for table in source['tables']}
    for event in events:
        table = tables[event['table']]
        rows = [row for row in candidate[table['id']]
                if canonical(row[table['primary_key']]) == canonical(event['key'])]
        require(len(rows) == 1, 'event target must identify exactly one row')
        rows[0].update(copy.deepcopy(event['set']))
    validate_data(source, candidate)
    return candidate
