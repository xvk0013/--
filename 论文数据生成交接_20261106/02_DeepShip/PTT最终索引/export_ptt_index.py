"""Export the PTT paper index without opening any waveform arrays."""

import csv
import json
from collections import Counter
from pathlib import Path


SPLITS = ('Train', 'Val', 'Test')
HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def export_index(parent_root, ptt_config_path, output_path):
    parent = read_json(parent_root / 'CONFIG.json')
    ptt = read_json(ptt_config_path)
    if parent['version'] != ptt['source_dataset_version']:
        raise ValueError('PTT must project the frozen SeenShip parent dataset.')
    if [c for c in parent['classes'] if c in ptt['class_names']] != ptt['class_names']:
        raise ValueError('PTT class order does not match the parent class order.')
    with (parent_root / 'DATA_INDEX.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)

    indices = {name: i for i, name in enumerate(ptt['set_names'])}
    selected = []
    for split in SPLITS:
        parent_rows = sorted((r for r in rows if r['split'] == split), key=lambda r: int(r['array_index']))
        kept = []
        for index, row in enumerate(parent_rows):
            name = row['label_name']
            if int(row['array_index']) != index or parent['set_names'][int(row['label_index'])] != name:
                raise ValueError(f'Parent index mismatch in {split}, row {index}.')
            if name not in indices:
                continue
            ptt_label = indices[name]
            if int(row['count']) != ptt['count_map'][ptt_label]:
                raise ValueError(f'PTT count mismatch in {split}, row {index}.')
            members = set(name.split('+'))
            target = [int(cls in members) for cls in ptt['class_names']]
            kept.append(dict(row, ptt_array_index=len(kept), ptt_label_index=ptt_label,
                             ptt_target=json.dumps(target, separators=(',', ':'))))
        if len(kept) != ptt['split_sizes'][split]:
            raise ValueError(f'Unexpected PTT size in {split}.')
        selected.extend(kept)

    with output_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ['ptt_array_index', 'ptt_label_index', 'ptt_target'])
        writer.writeheader()
        writer.writerows(selected)
    return selected


if __name__ == '__main__':
    parent_root = HERE.parent / 'Energy80_SeenShip父数据'
    config = HERE.parent / '读取协议' / 'Q0_DeepShip' / 'config_ptt_seen_ship.json'
    selected = export_index(parent_root, config, HERE / 'DATA_INDEX.csv')
    print(dict(Counter(row['split'] for row in selected)))
