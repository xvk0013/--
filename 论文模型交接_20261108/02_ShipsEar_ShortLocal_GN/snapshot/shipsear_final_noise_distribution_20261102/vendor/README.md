# Local model dependencies

These are byte-for-byte copies of the runtime sources used by retained V1.
They are included so this model does not import historical experiment modules.

- `tabm.py`: TabM 0.0.3, https://github.com/yandex-research/tabm,
  Apache License 2.0; full text in `LICENSE_TabM`. No algorithm modifications.
- `rtdl_num_embeddings.py`: 0.0.12, https://github.com/yandex-research/rtdl-num-embeddings,
  Apache License 2.0; full text in `LICENSE_rtdl_num_embeddings`. Required by TabM's
  imports; the retained model does not construct numerical embedding layers.
- `kymatio/`: Kymatio 0.3.0, https://github.com/kymatio/kymatio,
  BSD 3-Clause; full text in `LICENSE_Kymatio.md`. Original package copied unchanged,
  excluding generated `__pycache__`/`.pyc` files. Only its PyTorch 1D/2D transforms
  are used; alternative frontends are not activated.

Local source snapshot for all three: `D:/LSJ/Code/UART/shipsear/01_模型/当前主线_T1第27轮/vendor/`.

Project model extraction:
`CalibratedInteraction` and the waveform transforms are copied from that snapshot's
`model.py`; fixed 2D summary arithmetic comes from
`research/shipsear_fixed_spectral_factorial_20261031/model.py`, and the two-view
interface from `research/shipsear_f1m1_snr_balance_20261031/model.py`.
The seven-set plus count classifier and its state names retain the V1 path.
No neural encoder, adaptation checkpoint, or historical module import is needed.

Runtime environment: Python, PyTorch, NumPy, SciPy, packaging, typing_extensions.
The vendored rtdl module additionally attempts optional scikit-learn/tqdm imports.
Checkpoint files store the model/normalization/config separately; they are not
embedded in these source dependencies.
