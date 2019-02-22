#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2019 Unbabel <openkiwi@unbabel.com>
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

from kiwi.cli.better_argparse import ModelParser
from kiwi.cli.models.predictor_estimator import add_pretraining_options
from kiwi.models.predictor import Predictor


def parser_for_pipeline(pipeline):
    if pipeline == 'train':
        return ModelParser(
            'predictor',
            'train',
            title=Predictor.title,
            options_fn=add_pretraining_options,
            api_module=Predictor,
        )
    return None
