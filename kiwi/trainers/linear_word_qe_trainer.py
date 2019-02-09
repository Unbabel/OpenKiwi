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

from kiwi.models.linear.linear_trainer import LinearTrainer


class LinearWordQETrainer(LinearTrainer):
    def __init__(
        self, model, optimizer_name, regularization_constant, checkpointer
    ):
        super().__init__(
            classifier=model,
            checkpointer=checkpointer,
            algorithm=optimizer_name,
            regularization_constant=regularization_constant,
        )

    @property
    def model(self):
        return self.classifier
