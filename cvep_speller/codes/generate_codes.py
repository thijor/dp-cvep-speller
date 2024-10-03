import numpy as np
import pyntbci

# Shifted m-sequence
code = pyntbci.stimulus.make_m_sequence(
    poly=[1, 0, 0, 0, 0, 1], base=2, seed=6 * [1]
)[0, :]
codes = np.zeros((code.size, code.size), dtype="uint8")
for i in range(code.size):
    codes[i, :] = np.roll(code, i)
np.savez(file="mseq_61_shift.npz", codes=codes)  # [codes x bits]

# Original set of Gold codes
codes = pyntbci.stimulus.make_gold_codes(
    poly1=[1, 0, 0, 0, 0, 1], poly2=[1, 1, 0, 0, 1, 1], seed1=6 * [1], seed2=6 * [1]
)
np.savez(file="gold_61_6521.npz", codes=codes)  # [codes x bits]

# Modulated set of Gold codes
codes = pyntbci.stimulus.modulate(codes)
np.savez(file="mgold_61_6521.npz", codes=codes)  # [codes x bits]
