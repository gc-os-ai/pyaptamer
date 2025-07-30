from pyaptamer.datasets.loader import load_pfoa_structure
from pyaptamer.utils.struct_to_aaseq import struct_to_aaseq

print(struct_to_aaseq(load_pfoa_structure()))
