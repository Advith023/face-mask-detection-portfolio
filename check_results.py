"""Check published image-level results without model or ML dependencies."""
import csv
import re
from collections import Counter
from pathlib import Path


def check(root):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    with (root / 'results/kaggle-image-level.csv').open(newline='') as f:
        summary = list(csv.DictReader(f))
    configs = {r['setting'] for r in summary}
    labels = {'with_mask', 'without_mask'}
    groups = {(r['setting'], r['folder_label']) for r in summary}
    require(len(configs) == 3 and len(summary) == 6, 'Expected three configurations and six summary rows')
    require(groups == {(c, l) for c in configs for l in labels}, 'Invalid summary groups')
    counts = Counter()
    unreadable = Counter()
    seen = set()
    identities = {}
    paths = {}
    with (root / 'results/kaggle-per-image.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            digest, config = row['sha256'], row['configuration']
            label, prediction = row['folder_label'], row['prediction']
            path = row['relative_path']
            require(re.fullmatch('[0-9a-f]{64}', digest), 'Invalid SHA-256')
            require((config, label) in groups, 'Unknown configuration or label')
            require(prediction in labels | {'no_prediction'}, 'Invalid prediction')
            require(row['unreadable'] in {'True', 'False'}, 'Invalid unreadable flag')
            require((digest, config) not in seen, 'Duplicate image/configuration row')
            require(identities.get(digest, (label, path)) == (label, path), 'Image identity differs across configurations')
            require(paths.get(path, digest) == digest, 'Path has multiple image hashes')
            require(row['unreadable'] == 'False' or prediction == 'no_prediction', 'Unreadable image has a prediction')
            seen.add((digest, config))
            identities[digest] = (label, path)
            paths[path] = digest
            counts[config, label, prediction] += 1
            unreadable[config, label] += row['unreadable'] == 'True'
    require(len(identities) == 3748, 'Expected 3,748 unique evaluation images')
    require(len(seen) == len(identities) * len(configs), 'Missing image/configuration rows')
    for row in summary:
        config, label = row['setting'], row['folder_label']
        for column, prediction in [('predicted_mask', 'with_mask'), ('predicted_no_mask', 'without_mask'), ('no_prediction', 'no_prediction')]:
            require(counts[config, label, prediction] == int(row[column]), f'Mismatch: {config}/{label}/{column}')
        require(sum(counts[config, label, p] for p in labels | {'no_prediction'}) == int(row['images']), 'Image count mismatch')
        require(unreadable[config, label] == int(row['unreadable']), 'Unreadable count mismatch')
    print('PASS: 3,748 unique images, 11,244 predictions; all aggregate counts match.')


if __name__ == '__main__':
    check(Path(__file__).resolve().parent)
