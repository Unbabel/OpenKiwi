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
"""Use systems in a plugin way.

Solution based on https://julienharbulot.com/python-dynamical-import.html.
"""
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

# iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([package_dir]):
    # import the module and iterate through its attributes
    module = import_module(f"{__name__}.{module_name}")
    # for attribute_name in dir(module):
    #     attribute = getattr(module, attribute_name)
    #
    #     try:
    #         if isclass(attribute) and issubclass(attribute, QESystem):
    #             # Add the class to this package's variables
    #             globals()[attribute_name] = attribute
    #     except TypeError:
    #         pass
