import numpy as np
import pyntbci

if __name__ == "__main__":
    # Shifted m-sequence
    mseq = pyntbci.stimulus.make_m_sequence(
        poly=[1, 0, 0, 0, 0, 1],  # 6 1
        base=2,
        seed=6 * [1],
    )  # [1 x bits]
    mseqs = pyntbci.stimulus.shift(mseq, stride=1)  # [codes x bits]
    np.savetxt(
        fname="mseq_61_shift.txt", X=mseqs.astype("uint8"), fmt="%d", delimiter=","
    )

    # Set of Gold codes
    golds = pyntbci.stimulus.make_gold_codes(
        poly1=[1, 0, 0, 0, 0, 1],  # 6 1
        poly2=[1, 1, 0, 0, 1, 1],  # 6 5 2 1
        seed1=6 * [1],
        seed2=6 * [1],
    )  # [codes x bits]
    np.savetxt(
        fname="gold_61_6521.txt", X=golds.astype("uint8"), fmt="%d", delimiter=","
    )

    # Set of modulated Gold codes
    mgolds = pyntbci.stimulus.modulate(golds)  # [codes x bits]
    np.savetxt(
        fname="mgold_61_6521.txt", X=mgolds.astype("uint8"), fmt="%d", delimiter=","
    )
