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

from kiwi.utils.io import generate_slug, load_torch_file


def test_load_torch_file(model_dir):
    load_torch_file(model_dir / 'nuqe.ckpt')
    # There's no CUDA:
    with pytest.raises(AssertionError):
        load_torch_file(model_dir / 'nuqe.ckpt', map_location='cuda')
    # And this file does not exist:
    with pytest.raises(ValueError):
        load_torch_file(model_dir / 'nonexistent.torch')


def test_generate_slug():
    assert generate_slug('Some Random Text!') == 'some-random-text'
