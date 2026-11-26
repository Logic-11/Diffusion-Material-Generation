"""Run the predictor-corrector loop and convert final states to structures."""

from collections.abc import Mapping

import torch


class Sampler:
    def __init__(self, atom_pred, pos_pred, lat_pred, corrector, *, num_steps=None):
        """Use a separate step count when only the continuous fields are sampled."""
        if atom_pred is None and (type(num_steps) is not int or num_steps < 1):
            raise ValueError("Continuous sampling requires a positive integer num_steps")
        self.atom_pred = atom_pred
        self.num_steps = num_steps
        self.position_pred = pos_pred
        self.lattice_pred = lat_pred
        self.corrector = corrector

    @torch.no_grad()
    def sample_t(self, model, noisy, t, s):
        """Take one reverse step from t to s, with one time per crystal.

        Return (sample_state, mean_state). Atomic numbers are kept fixed when
        no atom predictor is supplied. Otherwise the mean contains posterior modes.
        """
        num_atoms, batch = noisy["num_atoms"], noisy["batch"]
        lattice = noisy["lattice"]
        t = torch.as_tensor(t, device=lattice.device)
        s = torch.as_tensor(s, device=lattice.device)
        if not t.is_floating_point():
            t = t.to(lattice.dtype)
        if (t.shape != num_atoms.shape or s.shape != num_atoms.shape
                or not torch.all((0 <= s) & (s < t) & (t <= 1))):
            raise ValueError("t and s must have shape [B] and satisfy 0 <= s < t <= 1")
        if self.atom_pred is not None:
            step_from_time = self.atom_pred.atom_corr.scheduler.step_from_time
            # s=0 is the clean endpoint; step_from_time itself numbers noisy levels from 1.
            step_s = torch.where(s == 0, 0, step_from_time(s))
            if not torch.all(step_from_time(t) == step_s + 1):
                raise ValueError("t and s must span exactly one reverse atom diffusion step")

        continuous_t, continuous_s = t.to(lattice), s.to(lattice)
        scores = self._predict_scores(model, noisy, continuous_t)
        corrected_coords = self.corrector.coord_correct(
            noisy["frac_coords"], scores["coord_score"], batch, num_atoms.numel())
        corrected_lattice = self.corrector.lattice_correct(
            lattice, scores["lattice_score"], continuous_t, continuous_s, self.lattice_pred.lat_corr)
        corrected = {**noisy, "frac_coords": corrected_coords, "lattice": corrected_lattice}

        # Predictors use the same corrected geometry and the updated scores at t.
        scores = self._predict_scores(model, corrected, continuous_t)
        sample_state, mean_state = corrected.copy(), corrected.copy()
        if self.atom_pred is not None:
            atom_sample, atom_mode = self.atom_pred.atom_type_predict(
                corrected["atomic_numbers"], scores["atom_logits"], t, batch)
            sample_state["atomic_numbers"] = atom_sample
            mean_state["atomic_numbers"] = atom_mode
        sample_state["frac_coords"], mean_state["frac_coords"] = self.position_pred.position_predict(
            corrected_coords, scores["coord_score"], continuous_t, continuous_s, num_atoms, batch)
        sample_state["lattice"], mean_state["lattice"] = self.lattice_pred.lattice_predict(
            corrected_lattice, scores["lattice_score"], continuous_t, continuous_s, num_atoms)
        return sample_state, mean_state

    @torch.no_grad()
    def sample(self, model, initial_state):
        """Denoise a batch of prior crystals from t=1 to 0 and return (sample, mean).

        Each step continues from the sampled state. The final mean contains the
        continuous predictor means and either fixed elements or posterior modes.
        Use ``to_structures(mean_state)`` to build the final crystals.
        """
        if self.atom_pred is None:
            steps = self.num_steps
        else:
            steps = model.timesteps
            if steps < 1 or steps != self.atom_pred.atom_corr.scheduler.timesteps:
                raise ValueError("model and atom predictor must share a positive number of diffusion steps")
        batch_shape = initial_state["num_atoms"].shape
        # Keep the discrete time grid precise. Continuous updates use the lattice dtype.
        times = torch.arange(steps, -1, -1, dtype=torch.float64,
                             device=initial_state["lattice"].device) / steps
        state = initial_state
        was_training = isinstance(model, torch.nn.Module) and model.training
        if was_training:
            model.eval()
        try:
            for index in range(steps):
                t = times[index].expand(batch_shape)
                s = times[index + 1].expand(batch_shape)
                state, mean_state = self.sample_t(model, state, t, s)
        finally:
            if was_training:
                model.train()
        return state, mean_state

    @staticmethod
    def to_structures(state: Mapping[str, torch.Tensor]):
        """Return a list of pymatgen Structures in batch order from one final state.

        Pass mean_state from sample for posterior-mode elements, or sample_state
        for sampled elements. Lattice vectors are rows in angstroms and positions
        are fractional. Site order within each crystal follows the input.
        """
        from pymatgen.core import Structure

        numbers = state["atomic_numbers"].detach().cpu()
        batch = state["batch"].detach().cpu()
        counts = state["num_atoms"].detach().cpu()
        coords = state["frac_coords"].detach().to(device="cpu", dtype=torch.float64)
        lattices = state["lattice"].detach().to(device="cpu", dtype=torch.float64)

        if any(field.dtype != torch.long for field in (numbers, batch, counts)):
            raise ValueError("atomic_numbers, batch and num_atoms must be int64")
        if (counts.ndim != 1 or counts.numel() == 0 or numbers.ndim != 1
                or batch.shape != numbers.shape or coords.shape != (numbers.numel(), 3)
                or lattices.shape != (counts.numel(), 3, 3)):
            raise ValueError("Expected one lattice per crystal and one fractional coordinate per atom")
        if (torch.any(counts <= 0) or torch.any(batch < 0) or torch.any(batch >= counts.numel())
                or not torch.equal(torch.bincount(batch, minlength=counts.numel()), counts)):
            raise ValueError("batch membership must agree with positive num_atoms")

        structures = []
        for index, lattice in enumerate(lattices):
            sites = batch == index
            species, frac = numbers[sites], coords[sites]
            invalid = species[(species < 1) | (species > 118)]
            if invalid.numel():
                raise ValueError(f"Crystal {index} contains atomic number {invalid[0].item()}. "
                                 "Expected a real element in 1..118 (0 is MASK).")
            if not torch.isfinite(frac).all() or not torch.isfinite(lattice).all():
                raise ValueError(f"Crystal {index} has non-finite coordinates or lattice")
            if torch.linalg.matrix_rank(lattice) < 3:
                raise ValueError(f"Crystal {index} has a singular lattice")
            structures.append(Structure(lattice.numpy(), species.tolist(), frac.numpy(),
                                        coords_are_cartesian=False, to_unit_cell=True))
        return structures

    @torch.no_grad()
    def _predict_scores(self, model, state, t):
        predict = getattr(model, "predict", None)
        scores_from_prediction = getattr(model, "scores_from_prediction", None)
        if not callable(predict) or not callable(scores_from_prediction):
            raise TypeError("model must provide callable predict and scores_from_prediction methods")
        for field in ("atomic_numbers", "frac_coords", "lattice", "num_atoms", "batch"):
            if field not in state:
                raise KeyError(f"state is missing required field {field!r}")
            if not torch.is_tensor(state[field]):
                raise TypeError(f"state[{field!r}] must be a torch.Tensor")

        prediction = predict(state, t)
        return scores_from_prediction(prediction, state, t)
