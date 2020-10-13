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
import pytest

from kiwi import constants as const
from kiwi.lib import search
from kiwi.lib.utils import save_config_to_file
#
#
# # TODO: can't import these...
# # from tests.test_nuqe import nuqe_config_dict, output_target_config
# @pytest.fixture
# def nuqe_config_dict(optimizer_config, data_processing_config):
#     encoder = dict(
#         window_size=3, embeddings=dict(source=dict(dim=50), target=dict(dim=50))
#     )
#     decoder_side = dict(hidden_sizes=[40, 20, 10, 5], dropout=0.0)
#     decoder = dict(target=decoder_side, source=decoder_side)
#     outputs = dict(
#         word_level=dict(
#             target=False,
#             gaps=False,
#             source=False,
#             class_weights=dict(
#                 target_tags={const.BAD: 3.0},
#                 gap_tags={const.BAD: 5.0},
#                 source_tags={const.BAD: 5.0},
#             ),
#         ),
#         sentence_level=dict(hter=False, use_distribution=False, binary=False),
#     )
#
#     config = dict(
#         class_name='NuQE',
#         batch_size=8,
#         num_data_workers=0,
#         model=dict(encoder=encoder, decoder=decoder, outputs=outputs),
#         optimizer=optimizer_config,
#         data_processing=data_processing_config,
#     )
#
#     return config
#
#
# @pytest.fixture
# def output_target_config(nuqe_config_dict):
#     nuqe_config_dict['model']['outputs']['word_level']['target'] = True
#     return nuqe_config_dict


def test_api(
    tmp_path, output_target_config, train_config, search_config, data_config, atol
):
    from kiwi.lib.search import search_from_file

    output_dir = tmp_path / 'search'

    search_config['directory'] = output_dir
    search_config['base_config']['data'] = data_config
    search_config['base_config']['system'] = output_target_config
    search_config['base_config']['trainer']['main_metric'] = 'target_tags_MCC'

    config_file = tmp_path / 'config.yaml'
    save_config_to_file(search.Configuration(**search_config), config_file)

    search_from_file(config_file)

    assert [file.name for file in output_dir.glob('*')] == ['0']
    assert set(file.name for file in (output_dir / '0').glob('*')) == set(
        [
            'parallel_coordinate.html',
            'optimization_history.html',
            'search_config.yaml',
            'study.pkl',
            'output.log',
        ]
    )

    search_from_file(config_file)
    assert set([file.name for file in output_dir.glob('*')]) == set(['0', '1'])


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__])  # pragma: no cover
