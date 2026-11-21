# v4: GemNet-T triplet diffusion

This version is based on the collaborator's updated `zqx` interface and adds
GemNet-T style one-hop three-body angular message passing.

## Main changes

- Continuous diffusion time `t in [1e-5, 1]`.
- MASK-first atom classes: `0=MASK, 1..118=atomic numbers`.
- Periodic graph image cap `max_cell_images_per_dim`.
- MP-20 cache under `data_cache/mp_20`.
- Unified `scripts/train.py`, `src/sampling.py`, and `scripts/generate.py`.
- GemNet-T triplet interaction:
  - build edge pairs sharing one center atom;
  - compute `cos(phi)`;
  - expand it with zero-order spherical harmonics / Legendre polynomials;
  - update edge features with angular messages.

The angular implementation is intentionally one-hop and triplet-only. It does
not include GemNet-Q dihedral/quadruplet interactions.

## Tests

Run from this directory:

```powershell
python -B -m unittest discover -s test -v
```

Triplet-specific tests are in `test/test_triplet_integration.py`.

## Training

```powershell
pip install -r requirements-train.txt
python -B -m scripts.train `
  data.input=data_cache/mp_20 `
  trainer.max_steps=1 `
  trainer.device=cpu
```

Disable angular interactions for an ablation:

```powershell
python -B -m scripts.train `
  data.input=data_cache/mp_20 `
  trainer.max_steps=1 `
  trainer.device=cpu `
  model.denoiser.use_triplets=false
```

## Generation

```powershell
python -B -m scripts.generate `
  --checkpoint .\output\best.pt `
  --num-samples 10 `
  --batch-size 2 `
  --device cpu
```

Checkpoints created before triplet support are still loadable with triplet
interactions disabled automatically.

## Relevant files

- `src/geometry.py`: periodic graph and triplet pair construction.
- `src/gnn.py`: Legendre cosine basis and `TripletInteraction`.
- `src/denoiser.py`: passes triplet inputs through every message-passing layer.
- `src/diffusion.py`: rebuilds triplet geometry from noisy coordinates.
- `scripts/train.py`: training and checkpoint configuration.
- `scripts/generate.py`: checkpoint loading and CIF generation.
