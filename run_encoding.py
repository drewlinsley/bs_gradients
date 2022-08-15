from brainscore import score_model
from model_tools.brain_transformation import LayerMappedModel, TemporalIgnore
from candidate_models.base_models import base_model_pool
import argparse
import pandas as pd
import os
import json
import pickle
import time
import sys
from utils_lore import boolify, dict_hash
import torch

os.environ["RESULTCACHING_DISABLE"] = '1'

# ======================
# Command line arguments
# ======================
MAIN_OPTS = ['gpu_id', 'version', 'benchmark_identifier', 'model_identifier']

# Parse arguments based on --argument=value pattern
opts = sys.argv[1:]

for opt in opts:
    assert (opt.startswith('--'))
    assert ('=' in opt)

opts = {x.split('=')[0][2:]: x.split('=')[1] for x in opts}
opts = {k: boolify(v) for k, v in opts.items()}  # if it looks like a boolean, turn it into a boolean

# Defaults
opts['gpu_id'] = opts.get('gpu_id', -1)
opts['version'] = opts.get('version', 'debug')
opts['parent_folder'] = opts.get('parent_folder', './data/splits/')
opts['baseline_splits'] = opts.get('baseline_splits', 10)
opts['csv_file'] = [opts.get('csv_file')] if opts.get('csv_file') else ['__majajhonglocal_halves_ty_0.1_neg.csv',
                                                                        '__majajhonglocal_halves_ty_0.1_pos.csv',
                                                                        '__majajhonglocal_halves_tz_0.1_neg.csv',
                                                                        '__majajhonglocal_halves_tz_0.1_pos.csv',
                                                                        '__majajhonglocal_halves_s_0.1_neg.csv',
                                                                        '__majajhonglocal_halves_s_0.1_pos.csv']

# Filter out main options that are only used here
encoding_kwargs = {k: v for k, v in opts.items() if k not in MAIN_OPTS}

# Hash the dictionary of options
opts_hash = dict_hash(opts)

print(opts)

# ======================
# Setting up
# ======================
# Time stamp
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Choose GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(opts['gpu_id'])
print(torch.cuda.is_available())


# Load info about which layer to select
with open("modelmeta-2019.json", 'r') as f:
    model_meta = json.load(f)
layer = [i['fields']['value'] for i in model_meta if
         i['model'] == 'benchmarks.ModelMeta' and
         i['fields']['key'] == 'IT_layer' and
         i['fields']['model'] == opts['model_identifier']]
assert (len(layer) == 1)
layer = layer[0]

# For result saving
result_dir = os.path.join('results', 'encoding_' + opts['version'])
os.makedirs(result_dir, exist_ok=True)
result_fname = os.path.join(result_dir, '__'.join([opts['model_identifier'], opts['benchmark_identifier'], opts_hash])+'.csv')

explained_variance_dir = os.path.join('results', 'explained_variance_' + opts['version'])
os.makedirs(explained_variance_dir, exist_ok=True)
explained_variance_fname = os.path.join(explained_variance_dir, '__'.join([opts['model_identifier'], opts['benchmark_identifier'], opts_hash])+'.csv')
encoding_kwargs['explained_variance_fname'] = explained_variance_fname

if opts['version'] != 'debug' and os.path.isfile(result_fname):
    raise FileExistsError('results file already exists')

# ======================
# Model and layer
# ======================

# layer -> region
model = base_model_pool[opts['model_identifier']]

# layer -> region
model = LayerMappedModel(opts['model_identifier'] + "__" + layer, activations_model=model, visual_degrees=8)

model.commit("IT", layer)

# ignore time_bins
model = TemporalIgnore(model)

print(model.identifier)


# ======================
# Evaluate all splits
# ======================

for csv_file in opts['csv_file']:
    # If baseline, match train_size and test_size to csv_file
    # If NO_CSV_FILE use the dicarlo default train_size and test_size
    if opts['baseline']:
        if csv_file == 'NO_CSV_FILE':
            encoding_kwargs['train_size'] = 0.9
            encoding_kwargs['test_size'] = 0.1
        else:
            train_csv = pd.read_csv(os.path.join(opts['parent_folder'], 'train' + csv_file), names=['path', 'id', 'cat', 'full'])
            test_csv = pd.read_csv(os.path.join(opts['parent_folder'], 'test' + csv_file), names=['path', 'id', 'cat', 'full'])
            encoding_kwargs['train_size'] = len(train_csv)
            encoding_kwargs['test_size'] = len(test_csv)
        encoding_kwargs['baseline_splits'] = opts['baseline_splits']
    else:
        if csv_file == 'NO_CSV_FILE':
            raise ValueError("Can't have NO_CSV_FILE if not running baseline")
    encoding_kwargs['csv_file'] = csv_file

    # Run it through the brainscore pipeline
    print(model.identifier, opts['benchmark_identifier'], encoding_kwargs)
    score = score_model(model_identifier=model.identifier, model=model,
                        benchmark_identifier=opts['benchmark_identifier'], **encoding_kwargs)

    # Gather and save results
    results = opts
    results['csv_file'] = csv_file
    results['score'] = score.values[0]
    results['score_std'] = score.values[0]
    results['similarity_mean'] = score.raw[0].item()
    results['similarity_std'] = score.raw[1].item()
    results['n_splits'] = score.raw.raw.shape[0]
    results['per_neuroid'] = [score.raw.raw.mean(dim='split').values]
    results_df = pd.DataFrame(results, index=[0])

    with open(result_fname, 'a') as f:
        results_df.to_csv(f, mode='a', header=f.tell() == 0)

    print(score)
