# Out-of-Distribution Transfer Test for Algorithmic Common Information (Δ_MDL) in Gene Regulatory Networks

Evaluates out-of-regime MDL compression transfer (`Δ_MDL`) in the Cho et al. (2003) ERK/RKIP pathway (`BIOMD0000000647`) against 128 topology-preserving randomized controls.

The theory behind Δ — the Gács–Körner three-part code, the Gács–Tromp–Vitányi two-part code, the Janzing–Schölkopf causal reading — is in [theoretical_foundations_of_delta.txt](theoretical_foundations_of_delta.txt), with the full reference list at the end of it.

## Usage
Requires Python 3 and `numpy`:
```bash
python3 -m pip install numpy
python3 transfer_647.py
```

It takes about 20–30 seconds. Expected output is in `transfer_647_example_output.txt`.

## Results in brief

- The biological network transfers: Δ = 11.297 within-regime, rising to 13.097 on the weak/shifted regime and 13.258 on random pulses.
- Shuffled-rate twins collapse on transfer: they beat bio 6/64 within-regime, but only 1/64 on the weak/shifted regime and 0/64 on random pulses.
- Uniform-rate twins are mixed: 3/64 beat bio on the amplitude shift, but 27/64 beat bio on the timing-only shift. What counts as a new environment — amplitude shift vs timing shift — is the open question.
