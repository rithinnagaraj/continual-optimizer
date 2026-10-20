"""Validate a completed, architecture-specific pilot selection (stdlib only)."""
import hashlib
import json
import math

def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def winner(rows):
    good=[r for r in rows if r['status']=='ok' and math.isfinite(r['validation_loss'])
          and math.isfinite(r['validation_accuracy']) and 0 <= r['validation_accuracy'] <= 1]
    if not good:
        raise ValueError('No finite candidate completed')
    return min(good,key=lambda r:(-r['validation_accuracy'],r['validation_loss'],r['lr']))['lr']

def load_selection(path, model, patch_size, source, baseline):
    payload=json.loads(path.read_text(encoding='utf-8'))
    cfg=payload['config']
    if payload.get('schema')!='cifar-cap-lr-selection-v1' or payload.get('config_hash')!=signature(cfg):
        raise ValueError('Invalid LR selection record')
    if cfg['source_version']!=source or cfg['patch_size']!=patch_size:
        raise ValueError('LR selection source or patch size differs; rerun the pilot for this protocol')
    if cfg['cap_without_lr']!={k:v for k,v in baseline.items() if k!='lr'}:
        raise ValueError('Non-LR CAP settings changed')
    rates=cfg['learning_rates']
    if len(rates)!=10 or len(set(rates))!=10 or any(not math.isfinite(x) or x<=0 for x in rates):
        raise ValueError('Selection requires ten distinct positive rates')
    rows=payload['results'][model]
    if len(rows)!=10 or sorted(r['lr'] for r in rows)!=sorted(rates):
        raise ValueError('Incomplete LR sweep')
    rate=winner(rows)
    if payload['selected'][model]!=rate:
        raise ValueError('Recorded winner disagrees with validation results')
    return rate,hashlib.sha256(path.read_bytes()).hexdigest()
