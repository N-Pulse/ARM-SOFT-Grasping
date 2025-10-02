import yaml, json, random
from pathlib import Path
from utils import split_train_val

ROOT = Path('dataset')                

def to_str(paths):                     
    return [str(p) for p in paths]

def collect():
    meta = {}
    meta['wake']    = to_str(sorted((ROOT/'wake_word').glob('*.wav')))
    meta['garbage'] = to_str(sorted((ROOT/'garbage').glob('*.wav')))

    cmds = {}
    for p in (ROOT/'commands').iterdir():
        if p.is_dir():
            cmds[p.name] = to_str(sorted(p.glob('*.wav')))
    meta['commands'] = cmds

    # train / val split
    train = {}; val = {}
    train['wake'],    val['wake']    = split_train_val(meta['wake'])
    train['garbage'], val['garbage'] = split_train_val(meta['garbage'])
    train['commands'] = {}; val['commands'] = {}
    for k, files in meta['commands'].items():
        tr, va = split_train_val(files)
        train['commands'][k] = tr
        val['commands'][k]   = va

    with open('meta_train.yaml', 'w') as f:
        yaml.safe_dump(train, f, allow_unicode=True)
    with open('meta_val.yaml', 'w') as f:
        yaml.safe_dump(val,   f, allow_unicode=True)

    print('✅  meta_train.yaml et meta_val.yaml créés.')

if __name__ == '__main__':
    collect()
