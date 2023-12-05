from collections import namedtuple

__PersuasivenessParams = namedtuple('PersuasivenessParams', ['name', 'mu', 'theta'])
Guillible = __PersuasivenessParams('Guillible', 0.5, 0.5)
Adamant = __PersuasivenessParams('Adamant', 0.25, 0.25)
Tolerant = __PersuasivenessParams('Tolerant', 0.25, 0.5)
Impressionable = __PersuasivenessParams('Impressionable', 0.5, 0.25)