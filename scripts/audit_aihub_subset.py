"""Describe selected AI Hub labels and render original/mask inspection pairs (CPU only)."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extracted', required=True, type=Path)
    p.add_argument('--prepared', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if not (a.extracted/'EXTRACTION_COMPLETE.json').is_file():
        raise ValueError('Extraction must be complete')
    report = json.loads((a.prepared/'report.json').read_text(encoding='utf-8'))
    if not report['complete'] or not (a.prepared/'.complete').is_file() or (a.prepared/'.incomplete').exists():
        raise ValueError('Native preparation must be complete')
    report_sha = hashlib.sha256((a.prepared/'report.json').read_bytes()).hexdigest()
    if json.loads((a.prepared/'.complete').read_text(encoding='utf-8'))['report_sha256'] != report_sha:
        raise ValueError('Native completion marker does not bind the report')
    a.output.mkdir(parents=True, exist_ok=False)
    source12 = next(a.extracted.glob('12.*'))
    shapes, labels, views = Counter(), Counter(), Counter()
    counts = Counter()
    groups = defaultdict(set)
    frame_names = defaultdict(set)
    for path in source12.rglob('*.json'):
        if '01.원천데이터' not in path.parts or path.name == 'EXTRACTION_COMPLETE.json':
            continue
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
        if 'shapes' not in doc:
            continue
        split = 'train' if 'Training' in path.parts else 'val'
        counts[split] += 1
        fields = path.stem.split('_')
        if len(fields) < 5:
            raise ValueError(f'Unexpected collection12 frame identity: {path.name}')
        groups[split].add('_'.join(fields[:-2]))
        views[fields[-2]] += 1
        frame_names[split].add(path.stem)
        for shape in doc['shapes']:
            shapes[shape['shape_type']] += 1
            labels[shape['label']] += 1
    buckets = defaultdict(list)
    for split in ('train', 'val'):
        for line in (a.prepared/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            buckets[(row['dataset_id'], split)].append(row)
    samples = []
    canvas = Image.new('RGB', (960, 214 * 12), '#151820')
    draw = ImageDraw.Draw(canvas)
    i = 0
    for key, rows in sorted(buckets.items()):
        # Include an explicitly annotated vulnerable-road-user/cone example if present.
        ranked = sorted(rows, key=lambda r: (-sum(r['source_class_counts'].get(c, 0)
                        for c in ('pedestrian', 'rider', 'bicycle', 'rubberCone')), r['sample_id']))
        selected = ranked[0] if ranked[0] != rows[0] else rows[len(rows)//2]
        for row in (rows[0], selected):
            with Image.open(row['image']) as img:
                rgb = img.convert('RGB').resize((480, 180), Image.Resampling.BILINEAR)
            with Image.open(row['target']) as img:
                target = np.asarray(img.resize((480, 180), Image.Resampling.NEAREST))
            base = np.asarray(rgb).astype(np.float32)
            palette = np.zeros_like(base)
            palette[target == 0] = [30, 220, 100]
            palette[target == 1] = [245, 65, 60]
            palette[target == 255] = [115, 115, 115]
            overlay = Image.fromarray(np.uint8(base * .55 + palette * .45))
            y = i * 214
            draw.text((8, y + 2), f'{key[0]} {key[1]} | original / road green, nonroad red, ignore gray', fill='white')
            canvas.paste(rgb, (0, y + 25)); canvas.paste(overlay, (480, y + 25))
            samples.append({k: row[k] for k in ('dataset_id','split','sample_id','image','target','target_pixel_counts')})
            i += 1
    canvas.save(a.output/'native_mask_samples.jpg', quality=92)
    out = {'complete': True, 'training_or_gpu_used': False,
           'prepared_report_sha256': report_sha,
           'native_counts': {'/'.join(k):len(v) for k,v in buckets.items()},
           'target_pixel_counts':report['target_pixel_counts'],
           'class_inventory':report['class_inventory'], 'ignored_class_inventory':report['ignored_class_inventory'],
           'sample_pairs':samples,
           'dataset12':{'annotation_counts':dict(counts),'shape_types':dict(shapes),'labels':dict(labels),
                        'camera_codes':dict(views),'candidate_original_clip_overlap':len(groups['train'] & groups['val']),
                        'exact_frame_name_overlap':len(frame_names['train'] & frame_names['val']),
                        'candidate_group_definition':'filename without camera-code and frame-index suffix; heuristic, not verified recording provenance',
                        'dense_road_training_supported':False,'calibrated_eight_camera_data_confirmed':False,
                        'reason':'Sparse line/rectangle and relation/QA labels do not provide dense road supervision or measured vehicle control. No matching task adapter is present.'},
           'limits':['Partial polygon supervision; ignored pixels are not evaluated or assumed safe.',
                     'Official clips retained; same-day independence is not claimed.',
                     'Visual examples are qualitative checks, not general model accuracy.']}
    (a.output/'data_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'counts':out['native_counts'],'dataset12':out['dataset12']},ensure_ascii=False))


if __name__ == '__main__':
    main()
