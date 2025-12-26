#!/usr/bin/env python3
import argparse
import ast
import json
import math
from pathlib import Path


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def parse_log(log_path):
    log_path = Path(log_path)
    epochs = []
    loss = []
    beam_acc = []
    perfect = []
    acc_10perc = []

    for line in log_path.read_text().splitlines():
        if '__log__:' not in line:
            continue
        payload = line.split('__log__:', 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        epochs.append(_safe_float(data.get('epoch')))
        loss.append(_safe_float(data.get('valid_lattice_xe_loss')))
        beam_acc.append(_safe_float(data.get('valid_lattice_beam_acc')))
        perfect.append(_safe_float(data.get('valid_lattice_perfect')))

        percs = data.get('valid_lattice_percs_diff')
        if percs is None:
            acc_10perc.append(math.nan)
        else:
            try:
                percs = ast.literal_eval(percs)
                acc_10perc.append(_safe_float(percs[0]))
            except Exception:
                acc_10perc.append(math.nan)

    return {
        'epochs': epochs,
        'loss': loss,
        'beam_acc': beam_acc,
        'perfect': perfect,
        'acc_10perc': acc_10perc,
    }


def default_output_path(log_path, root_dir=None):
    log_path = Path(log_path)
    root = Path(root_dir) if root_dir else Path.cwd()
    exp_id = log_path.parent.name if log_path.parent else 'exp'
    exp_name = log_path.parent.parent.name if log_path.parent and log_path.parent.parent else 'run'
    safe_exp_name = exp_name.replace('/', '_').replace(chr(92), '_')
    safe_exp_id = exp_id.replace('/', '_').replace(chr(92), '_')
    return root / f'metrics_{safe_exp_name}_{safe_exp_id}.png'


def plot_metrics(log_path, output_path=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError('matplotlib is required to plot metrics. Install it to generate graphs.') from exc

    data = parse_log(log_path)
    if not data['epochs']:
        return None

    epochs = data['epochs']
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    axes[0].plot(epochs, data['loss'], marker='o', label='valid_xe_loss')
    axes[0].set_ylabel('valid_xe_loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, data['acc_10perc'], marker='o', label='<=0.1Q (%)')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('0.1Q accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path = Path(output_path) if output_path else default_output_path(log_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def extract_secret_from_log(log_path):
    """Try to extract the secret array from train.log (if present)."""
    secret = None
    lines = Path(log_path).read_text().splitlines()
    mid = False
    sec = ''
    for line in lines:
        if (mid is False) and (line.find('- secrets: ') > 0 or line.find('secrets: [array') > 0):
            if '- secrets: ' in line:
                sec = line.split('secrets: ')[1].rstrip()
            else:
                sec = line.split('secrets: ')[-1].rstrip()
            if sec.endswith('])]'):
                try:
                    sec_clean = sec.replace('array(', '').replace(')', '')
                    secret = ast.literal_eval(sec_clean)
                except Exception:
                    secret = None
                break
            else:
                mid = True
                continue
        elif mid:
            sec += line.rstrip()
            if line.rstrip().endswith('])]'):
                mid = False
                try:
                    sec_clean = sec.replace('array(', '').replace(')', '')
                    secret = ast.literal_eval(sec_clean)
                except Exception:
                    secret = None
                break
    return secret


def extract_recover_info(log_path):
    """Scan log for secret_found and recover methods."""
    secret_found = False
    recover_methods = []
    for line in Path(log_path).read_text().splitlines():
        if 'Found secret match' in line or 'Found secret match - ending experiment' in line or 'bits in secret' in line and 'have been recovered' in line:
            secret_found = True
        if 'Distinguisher' in line:
            if 'd' not in recover_methods:
                recover_methods.append('d')
        if 'K=' in line:
            # find integer after K=
            try:
                kval = int(line.split('K=')[-1].split()[0].strip())
                if kval not in recover_methods:
                    recover_methods.append(kval)
            except Exception:
                pass
        # older logs might show 'Found secret match - ending experiment.' etc
    return secret_found, recover_methods


def load_params_from_dir(exp_dir):
    """Load params dict from params.pkl or checkpoint.pth if available."""
    params = {}
    pkl = Path(exp_dir) / 'params.pkl'
    if pkl.exists():
        try:
            import pickle
            pk = pickle.load(open(pkl, 'rb'))
            params = pk.__dict__
            return params
        except Exception:
            pass
    # fallback to checkpoint.pth
    ck = Path(exp_dir) / 'checkpoint.pth'
    if ck.exists():
        try:
            import torch
            ckdict = torch.load(str(ck), map_location='cpu')
            if isinstance(ckdict, dict) and 'params' in ckdict:
                params = ckdict['params']
        except Exception:
            pass
    return params


def compute_adj_samples(params, last_epoch):
    # epoch_size default 300000
    epoch_size = params.get('epoch_size', 300000) if isinstance(params, dict) else 300000
    try:
        epoch_size = int(epoch_size)
    except Exception:
        epoch_size = 300000
    logSamples = math.log2(((last_epoch + 1) * epoch_size)) if last_epoch is not None else 0
    reuse = 1 if (isinstance(params, dict) and params.get('reuse', False) in [True, 'True', 1, '1']) else 0
    K = params.get('K', 1) if isinstance(params, dict) else 1
    times_reused = params.get('times_reused', 1) if isinstance(params, dict) else 1
    try:
        K = float(K)
        times_reused = float(times_reused)
        penalty = reuse * math.log2(max(1.0, K * times_reused))
    except Exception:
        penalty = 0
    adj = round(logSamples - penalty, 2)
    return adj


def analyze_experiment(log_path):
    """Return a dict with keys: hamming, adjSamples, secret_found, recover_method, best_acc"""
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    exp_dir = log_path.parent

    # parse metrics
    data = parse_log(log_path)
    # best_acc: max of beam_acc, acc_10perc, perfect, correct (?)
    best_vals = []
    best_vals.extend([x for x in data.get('beam_acc', []) if not math.isnan(x)])
    best_vals.extend([x for x in data.get('acc_10perc', []) if not math.isnan(x)])
    best_vals.extend([x for x in data.get('perfect', []) if not math.isnan(x)])
    best_acc = max(best_vals) if len(best_vals) > 0 else math.nan

    # last epoch
    epochs = [e for e in data.get('epochs', []) if not math.isnan(e)]
    last_epoch = int(max(epochs)) if epochs else 0

    # secret
    secret = extract_secret_from_log(log_path)
    hamming = int(sum(secret[0])) if (secret is not None and len(secret) > 0) else None

    # recover info
    secret_found, recover_methods = extract_recover_info(log_path)
    recover = recover_methods[0] if len(recover_methods) > 0 else ''

    # params and adjSamples
    params = load_params_from_dir(exp_dir)
    adj = compute_adj_samples(params, last_epoch)

    return {
        'hamming': hamming if hamming is not None else 0,
        'adjSamples': adj,
        'secret_found': bool(secret_found),
        'recover_method': recover,
        'best_acc': round(float(best_acc), 2) if not (best_acc is None or math.isnan(best_acc)) else math.nan,
    }


def main():
    parser = argparse.ArgumentParser(description='Plot SALSA metrics from train.log')
    parser.add_argument('--log', required=True, help='Path to train.log')
    parser.add_argument('--out', default=None, help='Output PNG path (default: metrics_<exp>_<id>.png)')
    parser.add_argument('--eval', action='store_true', help='Compute evaluation table (hamming, adjSamples, secret_found, recover_method, best_acc) from the log and params')
    parser.add_argument('--eval-out', default=None, help='Path to save evaluation CSV (one-row)')
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f'log not found: {log_path}')
        return 0

    saved = plot_metrics(log_path, args.out)
    if saved:
        print(f'saved {saved}')
    else:
        print('no metrics found in log')

    if args.eval:
        eval_row = analyze_experiment(log_path)
        # try to pretty print using tabulate/pandas
        try:
            from tabulate import tabulate
            print('\nEvaluation:')
            print(tabulate([eval_row], headers='keys', tablefmt='grid', showindex=True))
        except Exception:
            print(eval_row)

        if args.eval_out:
            try:
                import csv
                with open(args.eval_out, 'w', newline='') as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(eval_row.keys()))
                    writer.writeheader()
                    writer.writerow(eval_row)
                print(f'Wrote evaluation CSV to {args.eval_out}')
            except Exception as exc:
                print('Failed to write eval csv:', exc)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
