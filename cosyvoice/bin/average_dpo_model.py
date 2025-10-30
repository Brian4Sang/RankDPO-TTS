# average_model.py
import os
import re
import argparse
import glob
import yaml
import torch

def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model path to save')
    parser.add_argument('--src_path', required=True, help='directory containing epoch_x_* checkpoints and yaml logs')
    parser.add_argument('--num', default=5, type=int, help='number of checkpoints to average')
    parser.add_argument('--select_mode',
                        default='last',
                        choices=['last', 'val_loss', 'dpo', 'rankdpo'],
                        help='model selection strategy')
    return parser.parse_args()

_EPOCH_RE = re.compile(r'epoch[_\-](\d+)')

def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default



def _read_yaml(path):
    # 读取一次文本，避免多次 file.seek
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # 1) 先尝试 safe_load（最快、最安全）
    try:
        data = yaml.safe_load(txt)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2) 回退到 BaseLoader（把所有标量当作纯文本/基础类型来解析，
    #    会忽略诸如 python/object/apply 之类的复杂 tag）
    try:
        data = yaml.load(txt, Loader=yaml.BaseLoader)
        return data if isinstance(data, dict) else {}
    except Exception:
        # 兜底：返回空 dict，后续逻辑会跳过该文件
        return {}

def _get_epoch(dic, yaml_path):
    ep = dic.get('epoch', None)
    if isinstance(ep, (int, float)) or (isinstance(ep, str) and str(ep).isdigit()):
        return int(ep)
    m = _EPOCH_RE.search(os.path.basename(yaml_path))
    return int(m.group(1)) if m else -1

def _dig_metrics(dic):
    """统一从 loss_dict 读取；缺失则回退到顶层"""
    ld = dic.get('loss_dict', {}) or {}
    Z = {}
    def getk(k):
        return _to_float(ld.get(k, dic.get(k)))
    Z['loss']      = getk('loss')
    Z['sft_loss']  = getk('sft_loss')
    Z['pair_acc']  = getk('pair_acc') or getk('dpo_acc') or getk('acc') # pair_acc 优先，无则退 acc
    Z['rank_acc']  = getk('rank_acc')
    Z['rewards']   = getk('rewards')
    Z['dpo_loss']  = getk('dpo_loss')
    Z['logps']     = getk('logps')
    Z['step']      = dic.get('step', 0)
    Z['tag']       = dic.get('tag', '')
    return Z

def collect_checkpoints(src_path):
    yamls = glob.glob(os.path.join(src_path, '*.yaml'))
    yamls = [f for f in yamls
             if not (os.path.basename(f).startswith('train')
                     or os.path.basename(f).startswith('init'))]
    info_list = []
    for y in yamls:
        dic = _read_yaml(y)
        epoch = _get_epoch(dic, y)
        met = _dig_metrics(dic)
        info = {
            'yaml': y,
            'epoch': epoch,
            'step': int(met['step'] or 0),
            'loss': _to_float(met['loss'], 1e9),
            'sft_loss': _to_float(met['sft_loss']),
            'pair_acc': _to_float(met['pair_acc']),
            'rank_acc': _to_float(met['rank_acc']),
            'rewards': _to_float(met['rewards']),
            'dpo_loss': _to_float(met['dpo_loss']),
            'logps': _to_float(met['logps']),
            'tag': met.get('tag', ''),
        }
        info_list.append(info)
    info_list.sort(key=lambda x: (x['epoch'], x['step']))
    return info_list

def _find_ckpt_by_epoch(src_path, epoch):
    patterns = [
        os.path.join(src_path, f'epoch_{epoch}_*.pt'),
        os.path.join(src_path, f'epoch-{epoch}-*.pt'),
        os.path.join(src_path, f'*epoch_{epoch}*.pt'),
        os.path.join(src_path, f'*epoch-{epoch}*.pt'),
    ]
    for p in patterns:
        matches = sorted(glob.glob(p))
        if matches:
            whole = [m for m in matches if 'whole' in os.path.basename(m)]
            return whole[0] if whole else matches[0]
    return None

def select_checkpoints(info_list, src_path, num, mode):
    filtered = [x for x in info_list if x['epoch'] is not None and x['epoch'] >= 0]
    if not filtered:
        raise RuntimeError('No valid YAML with epoch_x found.')

    if mode == 'last':
        selected = filtered[-num:]

    elif mode == 'val_loss':
        selected = sorted(filtered, key=lambda x: (_to_float(x['loss'], 1e9), x['epoch']))[:num]

    elif mode == 'dpo':
        # 主：pair_acc（或 acc）；副：dpo_loss↓、loss↓、rewards↑
        def key(x):
            dpo_acc = _to_float(x['pair_acc'], float('-inf'))  # 没有 pair_acc 则已在 collect 中回退为 acc
            dpo_loss = _to_float(x['dpo_loss'])
            loss = _to_float(x['loss'])
            rewards = _to_float(x['rewards'])
            return (
                dpo_acc,
                -(dpo_loss if dpo_loss is not None else 0.0),
                -(loss if loss is not None else 0.0),
                (rewards if rewards is not None else 0.0),
                x['epoch'],
            )
        selected = sorted(filtered, key=key, reverse=True)[:num]

    elif mode == 'rankdpo':
        # ✅ 主指标：pair_acc + rank_acc；次指标：dpo_loss（小优先）、loss（小优先）
        def key(x):
            pair_acc = _to_float(x['pair_acc'], 0.0)
            rank_acc = _to_float(x['rank_acc'], 0.0)
            primary = pair_acc + rank_acc
            dpo_loss = _to_float(x['dpo_loss'])
            loss = _to_float(x['loss'])
            return (
                primary,
                -(dpo_loss if dpo_loss is not None else 0.0),
                -(loss if loss is not None else 0.0),
                x['epoch'],
            )
        selected = sorted(filtered, key=key, reverse=True)[:num]

    else:
        raise ValueError(f'Unknown select_mode: {mode}')

    print('Selected (epoch, step, pair_acc, rank_acc, loss, dpo_loss):')
    for s in selected:
        print((s['epoch'], s['step'], s['pair_acc'], s['rank_acc'], s['loss'], s['dpo_loss']))

    path_list = []
    for s in selected:
        ckpt = _find_ckpt_by_epoch(src_path, s['epoch'])
        if ckpt is None:
            raise FileNotFoundError(f'Cannot find checkpoint for epoch {s["epoch"]} under {src_path}')
        path_list.append(ckpt)
    return path_list

def average_checkpoints(path_list, dst_model):
    print('Averaging checkpoints:')
    for p in path_list:
        print('  -', os.path.basename(p))
    avg = {}
    num = len(path_list)
    assert num > 0
    for path in path_list:
        states = torch.load(path, map_location='cpu')
        state_dict = states.get('state_dict') if isinstance(states, dict) else None
        if state_dict is None:
            state_dict = states
        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            if k not in avg:
                avg[k] = v.clone()
            else:
                avg[k] += v
    for k in avg.keys():
        avg[k] = torch.true_divide(avg[k], num)
    print('Saving to:', dst_model)
    torch.save(avg, dst_model)

def main():
    args = get_args()
    info_list = collect_checkpoints(args.src_path)
    path_list = select_checkpoints(info_list, args.src_path, args.num, args.select_mode)
    average_checkpoints(path_list, args.dst_model)

if __name__ == '__main__':
    main()