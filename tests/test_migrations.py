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

from kiwi.data.datasets.wmt_qe_dataset import WMTQEDataset
from kiwi.lib.predict import make_predictions


@pytest.mark.parametrize(
    'model_name,expected',
    [
        ('model-0.3.1.torch', 14.131476014852524),
        ('model-0.3.4.torch', 12.908155739307404),
    ],
)
def test_loading_old_model(
    tmp_path, data_config, model_dir, atol, model_name, expected
):
    load_model = model_dir / model_name

    predictions = make_predictions(
        output_dir=tmp_path,
        best_model_path=load_model,
        data_partition='valid',
        data_config=WMTQEDataset.Config(**data_config),
    )

    assert abs(sum(predictions['target_tags'][0]) - expected) < 0.1
