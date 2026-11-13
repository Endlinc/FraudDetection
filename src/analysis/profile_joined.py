"""Reproduce the full-data exploratory profile using only the standard library."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def profile():
    source = ROOT / 'data/train_joined.csv'
    with (ROOT / 'data/train_identity.csv').open(newline='') as stream:
        identity_ids = {r['TransactionID'] for r in csv.DictReader(stream)}
    groups = {k: defaultdict(lambda: [0, 0]) for k in ['Identity record', 'ProductCD', 'DeviceType', 'card4', 'card6', 'Amount band', '30-day period', 'Product × identity']}
    amounts = [[], []]
    seen = set()
    duplicate_ids = fraud = total = 0
    time_min, time_max = math.inf, -math.inf
    with source.open(newline='') as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        idx = {name: i for i, name in enumerate(columns)}
        missing = [0] * len(columns)
        missing_fraud = [0] * len(columns)
        for row in reader:
            assert len(row) == len(columns)
            key = row[idx['TransactionID']]
            duplicate_ids += key in seen
            seen.add(key)
            y = int(row[idx['isFraud']])
            assert y in (0, 1)
            total += 1
            fraud += y
            for i, value in enumerate(row):
                if value == '' or value.lower() == 'nan':
                    missing[i] += 1
                    missing_fraud[i] += y
            amt = float(row[idx['TransactionAmt']])
            dt = float(row[idx['TransactionDT']])
            assert math.isfinite(amt) and math.isfinite(dt)
            amounts[y].append(amt)
            time_min, time_max = min(time_min, dt), max(time_max, dt)
            identity = 'Present' if key in identity_ids else 'Absent'
            band = next((label for limit, label in [(25, '<25'), (50, '25–<50'), (100, '50–<100'), (200, '100–<200'), (500, '200–<500'), (1000, '500–<1,000')] if amt < limit), '≥1,000')
            values = {'Identity record': identity, 'Amount band': band, '30-day period': str(int(dt // (30 * 86400))), 'Product × identity': row[idx['ProductCD']] + ' / ' + identity}
            for field in ['ProductCD', 'DeviceType', 'card4', 'card6']:
                values[field] = row[idx[field]] or '(missing)'
            for field, value in values.items():
                groups[field][value][0] += 1
                groups[field][value][1] += y
    def quantiles(values):
        values.sort()
        def q(p):
            x = (len(values) - 1) * p
            lo, hi = math.floor(x), math.ceil(x)
            return values[lo] + (values[hi] - values[lo]) * (x-lo)
        return {'n': len(values), 'mean': sum(values)/len(values), 'min': values[0], 'p25': q(.25), 'median': q(.5), 'p75': q(.75), 'p95': q(.95), 'p99': q(.99), 'max': values[-1]}
    result = {'source': str(source), 'source_bytes': source.stat().st_size, 'rows': total, 'columns': len(columns), 'fraud': fraud, 'fraud_rate': fraud/total, 'duplicate_transaction_ids': duplicate_ids, 'time_seconds': [time_min, time_max], 'amount': {'Non-fraud': quantiles(amounts[0]), 'Fraud': quantiles(amounts[1]), 'All': quantiles(amounts[0]+amounts[1])}, 'groups': {}, 'missing': []}
    for field, mapping in groups.items():
        assert sum(v[0] for v in mapping.values()) == total
        assert sum(v[1] for v in mapping.values()) == fraud
        result['groups'][field] = [{'group': k, 'n': n, 'fraud': f, 'rate': f/n} for k, (n, f) in mapping.items()]
    for name, n, f in zip(columns, missing, missing_fraud):
        result['missing'].append({'column': name, 'missing': n, 'fraction': n/total, 'fraud_when_missing': f/n if n else None, 'fraud_when_present': (fraud-f)/(total-n) if total>n else None})
    result['checks'] = {'all_group_counts_reconcile': True, 'binary_labels': True, 'row_widths_valid': True, 'finite_amount_and_time': True}
    destination = ROOT / 'analysis/train_profile.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k,v in result.items() if k != 'missing'}, indent=2))
    print('Columns >50% missing:', sum(n/total > .5 for n in missing))
    print('Columns >90% missing:', sum(n/total > .9 for n in missing))

if __name__ == '__main__':
    profile()
