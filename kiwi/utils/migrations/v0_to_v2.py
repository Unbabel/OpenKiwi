#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2020 Unbabel <openkiwi@unbabel.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import sys
from collections import Counter
from typing import Any, Dict, List, Mapping, Tuple

import torch

import kiwi
from kiwi import constants as const
from kiwi.data.vocabulary import Vocabulary
from kiwi.utils.data_structures import DefaultFrozenDict

PREDICTOR_STATE_DICT_MAPPING = {
    'embeddings.target.embedding.weight': 'embedding_target.weight',
    'embeddings.source.embedding.weight': 'embedding_source.weight',
    'output_embeddings.target.weight': 'W1.weight',
    'encode_target.W2': 'W2',
    'encode_target.V': 'V',
    'encode_target.C': 'C',
    'encode_target.S': 'S',
    'encode_target.attention.scorer.layers.0.0.weight': 'attention.scorer.layers.0.0.weight',  # NOQA
    'encode_target.attention.scorer.layers.0.0.bias': 'attention.scorer.layers.0.0.bias',  # NOQA
    'encode_target.attention.scorer.layers.1.0.weight': 'attention.scorer.layers.1.0.weight',  # NOQA
    'encode_target.attention.scorer.layers.1.0.bias': 'attention.scorer.layers.1.0.bias',  # NOQA
}

PREDICTOR_STATE_DICT_DYNAMIC_MAPPING = {
    'encode_target.forward_backward_a.weight_ih_l{layer}': 'lstm_source.weight_ih_l{layer}',  # NOQA
    'encode_target.forward_backward_a.weight_hh_l{layer}': 'lstm_source.weight_hh_l{layer}',  # NOQA
    'encode_target.forward_backward_a.bias_ih_l{layer}': 'lstm_source.bias_ih_l{layer}',
    'encode_target.forward_backward_a.bias_hh_l{layer}': 'lstm_source.bias_hh_l{layer}',
    'encode_target.forward_backward_a.weight_ih_l{layer}_reverse': 'lstm_source.weight_ih_l{layer}_reverse',  # NOQA
    'encode_target.forward_backward_a.weight_hh_l{layer}_reverse': 'lstm_source.weight_hh_l{layer}_reverse',  # NOQA
    'encode_target.forward_backward_a.bias_ih_l{layer}_reverse': 'lstm_source.bias_ih_l{layer}_reverse',  # NOQA
    'encode_target.forward_backward_a.bias_hh_l{layer}_reverse': 'lstm_source.bias_hh_l{layer}_reverse',  # NOQA
    'encode_target.forward_b.weight_ih_l{layer}': 'forward_target.weight_ih_l{layer}',
    'encode_target.forward_b.weight_hh_l{layer}': 'forward_target.weight_hh_l{layer}',
    'encode_target.forward_b.bias_ih_l{layer}': 'forward_target.bias_ih_l{layer}',
    'encode_target.forward_b.bias_hh_l{layer}': 'forward_target.bias_hh_l{layer}',
    'encode_target.backward_b.weight_ih_l{layer}': 'backward_target.weight_ih_l{layer}',
    'encode_target.backward_b.weight_hh_l{layer}': 'backward_target.weight_hh_l{layer}',
    'encode_target.backward_b.bias_ih_l{layer}': 'backward_target.bias_ih_l{layer}',
    'encode_target.backward_b.bias_hh_l{layer}': 'backward_target.bias_hh_l{layer}',
}


def convert_state_dict_from_v0_to_v2_for_predictor(old_format_dict: Mapping) -> Dict:
    new_format_dict = {}

    # Predictor
    encoder_dict = {}
    for new_key, old_key in PREDICTOR_STATE_DICT_MAPPING.items():
        encoder_dict[new_key] = old_format_dict['Predictor']['state_dict'].pop(old_key)
    for new_key, old_key in PREDICTOR_STATE_DICT_DYNAMIC_MAPPING.items():
        layer = 0
        new_layer_key = new_key.format(layer=layer)
        old_layer_key = old_key.format(layer=layer)
        encoder_dict[new_layer_key] = old_format_dict['Predictor']['state_dict'].pop(
            old_layer_key
        )
        layer += 1
        while old_key.format(layer=layer) in old_format_dict['Predictor']['state_dict']:
            new_layer_key = new_key.format(layer=layer)
            old_layer_key = old_key.format(layer=layer)
            encoder_dict[new_layer_key] = old_format_dict['Predictor'][
                'state_dict'
            ].pop(old_layer_key)
            layer += 1

    out_embeddings_size = encoder_dict['output_embeddings.target.weight'].size(-1)
    encoder_dict['start_PreQEFV'] = torch.zeros(1, 1, out_embeddings_size)
    encoder_dict['end_PreQEFV'] = torch.zeros(1, 1, out_embeddings_size)

    new_format_dict['encoder'] = {
        '__version__': kiwi.__version__,
        'class_name': 'PredictorEncoder',
        'state_dict': encoder_dict,
    }

    new_format_dict['tlm_outputs'] = {
        '__version__': kiwi.__version__,
        'class_name': 'TLMOutputs',
        'state_dict': {},
    }

    # We should have converted all keys in the old state dict
    assert len(old_format_dict['Predictor']['state_dict'].keys()) == 0

    return new_format_dict


def convert_state_dict_from_v0_to_v2_for_estimator(old_format_dict: Mapping) -> Dict:
    new_format_dict = {}

    # Predictor
    encoder_dict = {}
    for new_key, old_key in PREDICTOR_STATE_DICT_MAPPING.items():
        encoder_dict[new_key] = old_format_dict['Estimator']['state_dict'].pop(
            f'predictor_tgt.{old_key}'
        )
    for new_key, old_key in PREDICTOR_STATE_DICT_DYNAMIC_MAPPING.items():
        old_key = f'predictor_tgt.{old_key}'
        layer = 0
        new_layer_key = new_key.format(layer=layer)
        old_layer_key = old_key.format(layer=layer)
        encoder_dict[new_layer_key] = old_format_dict['Estimator']['state_dict'].pop(
            old_layer_key
        )
        layer += 1
        while old_key.format(layer=layer) in old_format_dict['Estimator']['state_dict']:
            new_layer_key = new_key.format(layer=layer)
            old_layer_key = old_key.format(layer=layer)
            encoder_dict[new_layer_key] = old_format_dict['Estimator'][
                'state_dict'
            ].pop(old_layer_key)
            layer += 1

    out_embeddings_size = encoder_dict['output_embeddings.target.weight'].size(-1)
    encoder_dict['start_PreQEFV'] = torch.zeros(1, 1, out_embeddings_size)
    encoder_dict['end_PreQEFV'] = torch.zeros(1, 1, out_embeddings_size)

    new_format_dict['encoder'] = {
        '__version__': kiwi.__version__,
        'class_name': 'PredictorEncoder',
        'state_dict': encoder_dict,
    }

    # Estimator
    new_from_old_keys = {
        'mlp.0.weight': 'mlp.0.weight',
        'mlp.0.bias': 'mlp.0.bias',
        'lstm.weight_ih_l0': 'lstm.weight_ih_l0',
        'lstm.weight_hh_l0': 'lstm.weight_hh_l0',
        'lstm.bias_ih_l0': 'lstm.bias_ih_l0',
        'lstm.bias_hh_l0': 'lstm.bias_hh_l0',
        'lstm.weight_ih_l0_reverse': 'lstm.weight_ih_l0_reverse',
        'lstm.weight_hh_l0_reverse': 'lstm.weight_hh_l0_reverse',
        'lstm.bias_ih_l0_reverse': 'lstm.bias_ih_l0_reverse',
        'lstm.bias_hh_l0_reverse': 'lstm.bias_hh_l0_reverse',
    }
    decoder_dict = {}
    for new_key, old_key in new_from_old_keys.items():
        decoder_dict[new_key] = old_format_dict['Estimator']['state_dict'].pop(old_key)

    new_format_dict['decoder'] = {
        '__version__': kiwi.__version__,
        'class_name': 'EstimatorDecoder',
        'state_dict': decoder_dict,
    }

    # Outputs
    new_from_old_keys = {
        'word_outputs.target_tags.linear.weight': 'embedding_out.weight',
        'word_outputs.target_tags.linear.bias': 'embedding_out.bias',
        'word_outputs.target_tags.loss_fn.weight': 'xents.tags.weight',
        # 'word_outputs.gap_tags.linear.weight',
        # 'word_outputs.gap_tags.linear.bias',
        # 'word_outputs.gap_tags.loss_fn.weight',
        'sentence_outputs.sentence_scores.sentence_pred.linear_0.weight': 'sentence_pred.0.weight',  # NOQA
        'sentence_outputs.sentence_scores.sentence_pred.linear_0.bias': 'sentence_pred.0.bias',  # NOQA
        'sentence_outputs.sentence_scores.sentence_pred.linear_1.weight': 'sentence_pred.2.weight',  # NOQA
        'sentence_outputs.sentence_scores.sentence_pred.linear_1.bias': 'sentence_pred.2.bias',  # NOQA
        'sentence_outputs.sentence_scores.sentence_pred.linear_2.weight': 'sentence_pred.4.weight',  # NOQA
        'sentence_outputs.sentence_scores.sentence_pred.linear_2.bias': 'sentence_pred.4.bias',  # NOQA
        'sentence_outputs.sentence_scores.sentence_sigma.linear_0.weight': 'sentence_sigma.0.weight',  # NOQA
        'sentence_outputs.sentence_scores.sentence_sigma.linear_0.bias': 'sentence_sigma.0.bias',  # NOQA
        'sentence_outputs.sentence_scores.sentence_sigma.linear_1.weight': 'sentence_sigma.2.weight',  # NOQA
        'sentence_outputs.sentence_scores.sentence_sigma.linear_1.bias': 'sentence_sigma.2.bias',  # NOQA
        'sentence_outputs.sentence_scores.sentence_sigma.linear_2.weight': 'sentence_sigma.4.weight',  # NOQA
        'sentence_outputs.sentence_scores.sentence_sigma.linear_2.bias': 'sentence_sigma.4.bias',  # NOQA
        'sentence_outputs.binary.sentence_pred.linear_0.weight': 'binary_pred.0.weight',
        'sentence_outputs.binary.sentence_pred.linear_0.bias': 'binary_pred.0.bias',
        'sentence_outputs.binary.sentence_pred.linear_1.weight': 'binary_pred.2.weight',
        'sentence_outputs.binary.sentence_pred.linear_1.bias': 'binary_pred.2.bias',
        'sentence_outputs.binary.sentence_pred.linear_2.weight': 'binary_pred.4.weight',
        'sentence_outputs.binary.sentence_pred.linear_2.bias': 'binary_pred.4.bias',
    }

    outputs_dict = {}
    for new_key, old_key in new_from_old_keys.items():
        # Different output layers are optional
        if old_key in old_format_dict['Estimator']['state_dict']:
            outputs_dict[new_key] = old_format_dict['Estimator']['state_dict'].pop(
                old_key
            )

    new_format_dict['outputs'] = {
        '__version__': kiwi.__version__,
        'class_name': 'QEOutputs',
        'state_dict': outputs_dict,
    }

    new_format_dict['tlm_outputs'] = {
        '__version__': kiwi.__version__,
        'class_name': 'TLMOutputs',
        'state_dict': {},
    }

    # We should have converted all keys in the old state dict
    assert len(old_format_dict['Estimator']['state_dict'].keys()) == 0

    return new_format_dict


def convert_config_from_v0_to_v2_for_predictor(
    old_format_config: Dict[str, Any]
) -> Dict:
    old_format_config = dict(old_format_config)
    new_format_config = {
        'class_name': 'Predictor',
        'model': {
            'encoder': {
                'encode_source': old_format_config.pop('predict_inverse'),
                'hidden_size': old_format_config.pop('hidden_pred'),
                'rnn_layers': old_format_config.pop('rnn_layers_pred'),
                'dropout': old_format_config.pop('dropout_pred'),
                'share_embeddings': old_format_config.pop('share_embeddings'),
                'out_embeddings_dim': old_format_config.pop('out_embeddings_size'),
                'embeddings': {
                    'source': {'dim': old_format_config.pop('source_embeddings_size')},
                    'target': {'dim': old_format_config.pop('target_embeddings_size')},
                },
                'use_v0_buggy_strategy': False,  # Important
                'v0_start_stop': old_format_config.pop('start_stop', False),
            },
            'tlm_outputs': {'fine_tune': False},
        },
        'data_processing': {
            'vocab': {
                'max_size': {
                    'target': old_format_config.pop('target_vocab_size'),
                    'source': old_format_config.pop('source_vocab_size'),
                },
            },
        },
    }

    unmapped_old_config = {'source_side': 'source', 'target_side': 'target'}
    old_format_config.pop('__version__', None)  # Kiwi 0.3.4 already added this
    assert old_format_config.keys() == unmapped_old_config.keys()

    return new_format_config


def convert_config_from_v0_to_v2_for_estimator(
    old_format_config: Dict[str, Any]
) -> Dict:
    new_format_config = {
        'class_name': 'PredictorEstimator',
        'model': {
            'encoder': {
                'encode_source': old_format_config.pop('predict_inverse'),
                'hidden_size': old_format_config.pop('hidden_pred'),
                'rnn_layers': old_format_config.pop('rnn_layers_pred'),
                'dropout': old_format_config.pop('dropout_pred'),
                'share_embeddings': old_format_config.pop('share_embeddings'),
                'out_embeddings_dim': old_format_config.pop('out_embeddings_size'),
                'embeddings': {
                    'source': {'dim': old_format_config.pop('source_embeddings_size')},
                    'target': {'dim': old_format_config.pop('target_embeddings_size')},
                },
                'use_v0_buggy_strategy': True,  # Important
                'v0_start_stop': old_format_config['start_stop'],
            },
            'decoder': {
                'hidden_size': old_format_config.pop('hidden_est'),
                'rnn_layers': old_format_config.pop('rnn_layers_est'),
                'use_mlp': old_format_config.pop('mlp_est'),
                'dropout': old_format_config.pop('dropout_est'),
                'use_v0_buggy_strategy': True,  # Important
            },
            'outputs': {
                'word_level': {
                    'target': old_format_config.pop('predict_target'),
                    'gaps': old_format_config.pop('predict_gaps'),
                    'source': old_format_config.pop('predict_source'),
                    'class_weights': {
                        'target_tags': {
                            'BAD': old_format_config.pop('target_bad_weight')
                        },
                        'gap_tags': {'BAD': old_format_config.pop('gaps_bad_weight')},
                        'source_tags': {
                            'BAD': old_format_config.pop('source_bad_weight')
                        },
                    },
                },
                'sentence_level': {
                    'hter': old_format_config.pop('sentence_level'),
                    'use_distribution': old_format_config.pop('sentence_ll'),
                    'binary': old_format_config.pop('binary_level'),
                },
            },
            'tlm_outputs': {'fine_tune': old_format_config.pop('token_level')},
        },
        'data_processing': {
            # 'target_vocab_min_frequency': 1,
            # 'source_vocab_min_frequency': 1,
            'vocab': {
                'max_size': {
                    'target': old_format_config.pop('target_vocab_size'),
                    'source': old_format_config.pop('source_vocab_size'),
                },
            },
            # 'keep_rare_words_with_embeddings': False,
            # 'add_embeddings_vocab': False,
        },
    }

    unmapped_old_config = {
        'source_side': 'source',
        'target_side': 'target',
        'start_stop': False,
    }
    old_format_config.pop('__version__', None)  # Kiwi 0.3.4 already added this
    assert old_format_config.keys() == unmapped_old_config.keys()

    return new_format_config


def convert_vocab_v0_to_v2(old_vocabulary) -> Vocabulary:
    new_vocabulary = Vocabulary(counter=Counter())
    if const.UNK in old_vocabulary.stoi:
        new_vocabulary.unk_token = const.UNK
        new_vocabulary.specials.append(const.UNK)
    if const.PAD in old_vocabulary.stoi:
        new_vocabulary.pad_token = const.PAD
        new_vocabulary.specials.append(const.PAD)
    if const.START in old_vocabulary.stoi:
        new_vocabulary.bos_token = const.START
        new_vocabulary.specials.append(const.START)
    if const.STOP in old_vocabulary.stoi:
        new_vocabulary.eos_token = const.STOP
        new_vocabulary.specials.append(const.STOP)

    new_vocabulary.freqs = old_vocabulary.freqs
    # Remove some Nones in tag vocabularies
    while old_vocabulary.itos[-1] is None:
        old_vocabulary.itos.pop()
    new_vocabulary.itos = old_vocabulary.itos
    new_vocabulary.stoi = DefaultFrozenDict(
        {tok: i for i, tok in enumerate(new_vocabulary.itos)},
        default_key=new_vocabulary.unk_token,
    )

    return new_vocabulary


def convert_vocabs_from_v0_to_v2(old_format_vocabs: List[Tuple[str, Any]]) -> Dict:
    converted_vocabs = {k: convert_vocab_v0_to_v2(v) for k, v in old_format_vocabs}
    new_format_vocabs = {
        name: converted_vocabs[name]
        for name in [const.SOURCE, const.TARGET, const.PE]
        if name in converted_vocabs
    }
    if 'tags' in converted_vocabs:
        new_format_vocabs[const.TARGET_TAGS] = converted_vocabs['tags']
        new_format_vocabs[const.GAP_TAGS] = converted_vocabs['tags']
        new_format_vocabs[const.SOURCE_TAGS] = converted_vocabs['tags']

    return new_format_vocabs


def convert_trained_model_from_v0_to_v2(old_format_dict: Mapping) -> Dict:
    if (
        'Estimator' not in old_format_dict.keys()
        and 'Predictor' not in old_format_dict.keys()
    ):
        raise ValueError(
            'Loading old models only work for the Estimator (QE) model and the '
            'Predictor (TLM) model'
        )

    new_format_dict = {}

    if 'Predictor' in old_format_dict.keys():
        new_format_dict = {'__version__': kiwi.__version__, 'class_name': 'Predictor'}

        # Config
        new_format_config = convert_config_from_v0_to_v2_for_predictor(
            old_format_dict['Predictor']['config']
        )
        new_format_dict['config'] = new_format_config

        # Vocabularies
        new_format_vocabularies = convert_vocabs_from_v0_to_v2(old_format_dict['vocab'])
        new_format_dict[const.VOCAB] = new_format_vocabularies

        # Stated dicts
        new_format_state_dict = convert_state_dict_from_v0_to_v2_for_predictor(
            old_format_dict
        )
        new_format_dict.update(new_format_state_dict)

        # Copy sub-configs to sub-blocks
        new_format_dict['encoder']['config'] = new_format_config['model']['encoder']
        new_format_dict['tlm_outputs']['config'] = new_format_config['model'][
            'tlm_outputs'
        ]

    elif 'Estimator' in old_format_dict.keys():
        new_format_dict = {
            '__version__': kiwi.__version__,
            'class_name': 'PredictorEstimator',
        }

        # Config
        new_format_config = convert_config_from_v0_to_v2_for_estimator(
            old_format_dict['Estimator']['config']
        )
        new_format_dict['config'] = new_format_config

        # Vocabularies
        new_format_vocabularies = convert_vocabs_from_v0_to_v2(old_format_dict['vocab'])
        new_format_dict[const.VOCAB] = new_format_vocabularies

        # Stated dicts
        new_format_state_dict = convert_state_dict_from_v0_to_v2_for_estimator(
            old_format_dict
        )
        new_format_dict.update(new_format_state_dict)

        # Copy sub-configs to sub-blocks
        new_format_dict['encoder']['config'] = new_format_config['model']['encoder']
        new_format_dict['decoder']['config'] = new_format_config['model']['decoder']
        new_format_dict['outputs']['config'] = new_format_config['model']['outputs']
        new_format_dict['tlm_outputs']['config'] = new_format_config['model'][
            'tlm_outputs'
        ]

    return new_format_dict


if __name__ == '__main__':
    assert len(sys.argv) == 3
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    old_dict = torch.load(input_file, map_location='cpu')

    new_dict = convert_trained_model_from_v0_to_v2(old_dict)

    torch.save(new_dict, output_file)
