# average_model.py

import os
import argparse
import glob
import yaml
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')
    parser.add_argument('--select_mode',
                        default='last',
                        choices=['last', 'val_loss', 'dpo', 'rankdpo'],
                        help='model selection strategy')
    args = parser.parse_args()
    print(args)
    return args
def collect_checkpoints(src_path):
    """Collect candidate checkpoints and their yaml logs"""
    yamls = glob.glob(os.path.join(src_path, '*.yaml'))
    yamls = [
        f for f in yamls
        if not (os.path.basename(f).startswith('train')
                or os.path.basename(f).startswith('init'))
    ]
    info_list = []
    for y in yamls:
        with open(y, 'r') as f:
            dic_yaml = yaml.load(f, Loader=yaml.BaseLoader)
        epoch = int(dic_yaml['epoch'])
        step = int(dic_yaml.get('step', 0))
        tag = dic_yaml.get('tag', '')
        loss = float(dic_yaml['loss_dict'].get('loss', 1e9))

        # 常见指标
        dpo_acc = dic_yaml.get('dpo_acc')
        rank_acc = dic_yaml.get('rank_acc')
        chosen_logps = dic_yaml.get('chosen_logps')
        rejected_logps = dic_yaml.get('rejected_logps')
        chosen_reward = dic_yaml.get('chosen_reward')
        reject_reward = dic_yaml.get('reject_reward')
        sft_loss = dic_yaml.get('sft_loss')

        info_list.append({
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "dpo_acc": float(dpo_acc) if dpo_acc is not None else None,
            "rank_acc": float(rank_acc) if rank_acc is not None else None,
            "chosen_logps": float(chosen_logps) if chosen_logps is not None else None,
            "rejected_logps": float(rejected_logps) if rejected_logps is not None else None,
            "chosen_reward": float(chosen_reward) if chosen_reward is not None else None,
            "reject_reward": float(reject_reward) if reject_reward is not None else None,
            "sft_loss": float(sft_loss) if sft_loss is not None else None,
            "tag": tag,
        })
    return sorted(info_list, key=lambda x: x['epoch'])


def select_checkpoints(info_list, src_path, num, mode):
    """Return list of checkpoint paths according to mode"""
    if mode == 'last':
        selected = info_list[-num:]

    elif mode == 'val_loss':
        selected = sorted(info_list, key=lambda x: x['loss'])[:num]

    elif mode == 'dpo':
        def dpo_key(x):
            # 主指标 dpo_acc（越高越好）
            primary = x['dpo_acc'] if x['dpo_acc'] is not None else -1e9
            # 副指标 1: ΔlogP
            delta_logp = None
            if x['chosen_logps'] is not None and x['rejected_logps'] is not None:
                delta_logp = x['chosen_logps'] - x['rejected_logps']
            # 副指标 2: Δreward
            delta_reward = None
            if x['chosen_reward'] is not None and x['reject_reward'] is not None:
                delta_reward = x['chosen_reward'] - x['reject_reward']
            # 副指标 3: sft_loss（越低越好，所以取负）
            sft = -x['sft_loss'] if x['sft_loss'] is not None else 0
            return (primary, delta_logp or 0, delta_reward or 0, sft)

        selected = sorted(info_list, key=dpo_key, reverse=True)[:num]

    elif mode == 'rankdpo':
        def rankdpo_key(x):
            # 主指标 rank_acc（越高越好）
            primary = x['rank_acc'] if x['rank_acc'] is not None else -1e9
            # 副指标 loss（越低越好，所以取负）
            return (primary, -x['loss'])
        selected = sorted(info_list, key=rankdpo_key, reverse=True)[:num]

    else:
        raise ValueError(f"Unknown select_mode: {mode}")

    print("Selected checkpoints (epoch, step, loss, dpo_acc, rank_acc):")
    for s in selected:
        print(s)
    return [
        os.path.join(src_path, f'epoch_{s["epoch"]}_whole.pt') for s in selected
    ]

def average_checkpoints(path_list, dst_model):
    avg = {}
    num = len(path_list)
    assert num > 0
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        for k in states.keys():
            if k not in ['step', 'epoch']:
                if k not in avg.keys():
                    avg[k] = states[k].clone()
                else:
                    avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(dst_model))
    torch.save(avg, dst_model)


def main():
    args = get_args()
    info_list = collect_checkpoints(args.src_path)
    path_list = select_checkpoints(info_list, args.src_path, args.num, args.select_mode)
    average_checkpoints(path_list, args.dst_model)


if __name__ == '__main__':
    main()