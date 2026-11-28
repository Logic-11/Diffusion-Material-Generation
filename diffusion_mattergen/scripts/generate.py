import argparse
import json
import logging
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

if __name__ == "__main__" and not __package__:
    # Allow direct script execution to import src from the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backbone.denoiser import Denoiser
from src.csp import CSPDiffusion, CSPLoss, build_csp_prior, build_csp_sampler
from src.diffusion.checkpoint import checkpoint_task, read_model_checkpoint
from src.diffusion.corruption import AtomTypeCorruption, LatticeCorruption, PositionCorruption
from src.diffusion.crystal_diffusion import CrystalDiffusion
from src.diffusion.joint_diffusion import JointDiffusion
from src.diffusion.loss import JointDiffusionLoss
from src.diffusion.sampling.correctors import LangevinCorrector
from src.diffusion.sampling.pc_sampler import Sampler
from src.diffusion.sampling.predictors import AtomTypePredictor, LatticePredictor, PositionPredictor
from src.diffusion.sampling.prior import sample_n, sample_prior
from src.diffusion.scheduler import LatticeVPScheduler, MaskScheduler, PositionVEScheduler

if TYPE_CHECKING:
    from pymatgen.core import Structure

# MatterGen's SELECTED_ATOMIC_NUMBERS for unconditional generation.
MATTERGEN_ALLOWED_ATOMIC_NUMBERS = tuple(
    z for z in range(1, 84) if z not in {2, 10, 18, 36, 43, 54, 61}
)


def load_model(
    checkpoint_path: str | Path, device: str | torch.device = "cpu", *,
    expected_task: str | None = None,
) -> CrystalDiffusion:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use device='cpu' to load the model.")

    checkpoint = read_model_checkpoint(checkpoint_path)
    task = checkpoint_task(checkpoint)
    if expected_task is not None and task != expected_task:
        hint = (" Use --zero-shot with --mode csp to try joint weights."
                if expected_task == "csp" else "")
        raise ValueError(f"Expected a {expected_task} checkpoint, got {task}.{hint}")

    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]
    geometry = {
        "denoiser": Denoiser(**model_config),
        "position_corruption": PositionCorruption(PositionVEScheduler(**diffusion_config["position"])),
        "lattice_corruption": LatticeCorruption(LatticeVPScheduler(**diffusion_config["lattice"])),
        **diffusion_config["graph"],
    }
    if task == "csp":
        model = CSPDiffusion(loss_fn=CSPLoss(**diffusion_config["loss"]), **geometry)
    else:
        model = JointDiffusion(
            atom_corruption=AtomTypeCorruption(
                scheduler=MaskScheduler(timesteps=diffusion_config["timesteps"]),
                atom_types=model_config["atom_types"],
                mask_type=diffusion_config["input_mask_type"],
            ),
            loss_fn=JointDiffusionLoss(**diffusion_config["loss"]),
            **geometry,
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def generate_batch(
    model: CrystalDiffusion,
    sampler: Sampler,
    statistic_path: str | Path | None = None,
    batch_size: int = 1,
    *,
    composition: Mapping[str, int] | None = None,
    show_progress: bool = False,
    progress_desc: str = "Sampling",
    progress_position: int = 0,
) -> list["Structure"]:
    """Generate a batch from a fixed composition or the joint atom-count prior."""
    if composition is not None:
        initial_state = build_csp_prior(model, composition, batch_size)
    else:
        if statistic_path is None:
            raise ValueError("Joint generation requires statistic.json")
        sigma_max = model.position_corruption.scheduler.sigma_max
        lattice_corruption = model.lattice_corruption
        num_atoms = sample_n(statistic_path, batch_size=batch_size, device=sigma_max.device)
        atomic_numbers, frac_coords, lattice = sample_prior(
            num_atoms,
            limit_density=lattice_corruption.scheduler.limit_density.item(),
            sigma_max=sigma_max.item(),
            lat_cor=lattice_corruption,
            device=sigma_max.device,
        )
        initial_state = {
            "atomic_numbers": atomic_numbers,
            "frac_coords": frac_coords.to(lattice),
            "lattice": lattice,
            "num_atoms": num_atoms,
            "batch": torch.repeat_interleave(num_atoms, output_size=atomic_numbers.numel()),
        }
    _, mean_state = sampler.sample(
        model, initial_state, show_progress=show_progress,
        progress_desc=progress_desc, progress_position=progress_position,
    )
    return sampler.to_structures(mean_state)


def _generation_devices(device_name: str | None, num_devices: int, num_samples: int) -> tuple[torch.device, ...]:
    device = torch.device(device_name or ("cuda" if num_devices > 1 or torch.cuda.is_available() else "cpu"))
    if num_devices == 1:
        return (device,)
    if device.type != "cuda":
        raise ValueError("--devices greater than one requires --device cuda or cuda:N")

    first_device = 0 if device.index is None else device.index
    available = torch.cuda.device_count()
    if first_device + num_devices > available:
        raise ValueError(
            f"--devices {num_devices} from cuda:{first_device} requires "
            f"{first_device + num_devices} visible CUDA GPUs, but found {available}"
        )
    return tuple(torch.device("cuda", first_device + rank) for rank in range(min(num_devices, num_samples)))


def _cif_path(output_dir: Path, sample_index: int) -> Path:
    return output_dir / f"crystal_{sample_index:06d}.cif"


@contextmanager
def _generation_log(output_dir: Path):
    logger = logging.Logger(__name__, level=logging.INFO)
    handler = logging.FileHandler(output_dir / "generate.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        handler.close()


def _generate_cifs(args, statistic_path: Path | None, device: torch.device,
                   sample_indices: range, seed: int, *, progress_position: int = 1):
    expected_task = "joint" if args.zero_shot else args.mode
    model = load_model(args.checkpoint, device=device, expected_task=expected_task)
    if args.mode == "csp":
        sampler = build_csp_sampler(model)
    else:
        allowed_atomic_numbers = [
            z for z in MATTERGEN_ALLOWED_ATOMIC_NUMBERS if z <= model.atom_types
        ]
        sampler = Sampler(
            atom_pred=AtomTypePredictor(model.atom_corruption, allowed_atomic_numbers=allowed_atomic_numbers),
            pos_pred=PositionPredictor(model.position_corruption),
            lat_pred=LatticePredictor(model.lattice_corruption),
            corrector=LangevinCorrector(),
        )
    torch.manual_seed(seed)
    num_batches = (len(sample_indices) + args.batch_size - 1) // args.batch_size
    for start in range(0, len(sample_indices), args.batch_size):
        batch_indices = sample_indices[start:start + args.batch_size]
        batch_number = start // args.batch_size + 1
        structures = generate_batch(
            model, sampler, statistic_path, batch_size=len(batch_indices), composition=args.composition,
            show_progress=args.show_progress,
            progress_desc=f"Sampling ({device}, batch {batch_number}/{num_batches})",
            progress_position=progress_position,
        )
        for index, structure in zip(batch_indices, structures, strict=True):
            path = _cif_path(args.output, index)
            cif = structure.to(fmt="cif")
            output = path.open("x", encoding="utf-8")
            try:
                with output:
                    output.write(cif)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        yield len(batch_indices)


def _report_progress(logger, args, saved: int, *, device: torch.device | None = None) -> None:
    message = f"Saved {saved}/{args.num_samples} crystals to {args.output}"
    if device is not None:
        message += f" | device={device}"
    logger.info("%s", message)
    if not args.show_progress:
        print(message, flush=True)


def _generate_worker(rank, args, statistic_path: Path | None, devices: tuple[torch.device, ...], progress) -> None:
    # Share the console lock and give each GPU its own sampling progress row.
    tqdm.set_lock(progress.get_lock())
    device = devices[rank]
    if device.type == "cuda":
        torch.cuda.set_device(device)
    sample_indices = range(rank + 1, args.num_samples + 1, len(devices))
    seed = (args.seed + rank) % (1 << 64)
    with _generation_log(args.output) as logger:
        # Serialize log writes so the counts and messages share the same order.
        with progress.get_lock():
            logger.info(
                "Starting worker | device=%s | num_samples=%d | seed=%d", device, len(sample_indices), seed,
            )
        for batch_count in _generate_cifs(
            args, statistic_path, device, sample_indices, seed, progress_position=rank + 1,
        ):
            with progress.get_lock():
                progress.value += batch_count
                _report_progress(logger, args, progress.value, device=device)


def _parse_composition(value: str) -> dict[str, int]:
    message = 'Expected a JSON object with positive integer atom counts, e.g. {"Na":4,"Cl":4}'
    try:
        composition = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(message) from exc
    if (not isinstance(composition, dict) or not composition
            or any(type(count) is not int or count < 1 for count in composition.values())):
        raise argparse.ArgumentTypeError(message)
    return composition


def main(argv: list[str] | None = None) -> None:
    """Generate crystals in batches and write one CIF per structure."""
    parser = argparse.ArgumentParser(description="Generate crystal structures from a training checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint (.pt).")
    parser.add_argument("--mode", choices=("joint", "csp"), default="joint",
                        help="Generation mode. Defaults to joint diffusion.")
    parser.add_argument("--composition", type=_parse_composition,
                        help='Exact unit-cell composition for CSP, e.g. {"Na":4,"Cl":4}.')
    parser.add_argument("--zero-shot", action="store_true",
                        help="Use a joint checkpoint as a zero-shot CSP baseline.")
    parser.add_argument("--statistic", type=Path,
                        help="Path to statistic.json for joint generation. Defaults to the file beside the checkpoint.")
    parser.add_argument("-o", "--output", type=Path, default=Path("generated"),
                        help="Directory for CIF files and generate.log. Defaults to generated.")
    parser.add_argument("-n", "--num-samples", "--num_samples", type=int, default=64,
                        help="Total crystals to generate. Defaults to 64.")
    parser.add_argument("-b", "--batch-size", "--batch_size", type=int, default=64,
                        help="Maximum crystals per batch on each device. Defaults to 64.")
    parser.add_argument("--device", help="Torch device or first CUDA GPU, e.g. cpu or cuda:0. "
                                         "Defaults to CUDA if available, else CPU for single-device runs.")
    parser.add_argument("--devices", type=int, default=1,
                        help="Maximum CUDA GPUs to use, starting at --device. Defaults to 1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling. Defaults to 42.")
    parser.add_argument("--no-progress", action="store_false", dest="show_progress",
                        help="Disable generation and sampling progress bars.")
    args = parser.parse_args(argv)

    if args.num_samples < 1 or args.batch_size < 1:
        parser.error("--num-samples and --batch-size must both be positive")
    if args.devices < 1:
        parser.error("--devices must be positive")
    if not -(1 << 63) <= args.seed < (1 << 64):
        parser.error("--seed must fit the PyTorch seed range [-2**63, 2**64 - 1]")
    statistic_path = None
    if args.mode == "csp":
        if args.composition is None:
            parser.error("--mode csp requires --composition")
        if args.statistic is not None:
            parser.error("--statistic is only used in joint mode")
    else:
        if args.composition is not None or args.zero_shot:
            parser.error("--composition and --zero-shot require --mode csp")
        statistic_path = args.statistic or args.checkpoint.with_name("statistic.json")
    for path in (args.checkpoint, statistic_path):
        if path is not None and not path.is_file():
            parser.error(f"Expected an input file at {path}")

    try:
        devices = _generation_devices(args.device, args.devices, args.num_samples)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    sample_indices = range(1, args.num_samples + 1)
    for index in sample_indices:
        path = _cif_path(args.output, index)
        if path.exists():
            parser.error(f"{path} already exists. Choose a different --output directory.")

    with _generation_log(args.output) as logger:
        started = perf_counter()
        try:
            logger.info(
                "Starting generation | mode=%s | zero_shot=%s | composition=%s | checkpoint=%s | "
                "statistic=%s | output=%s | num_samples=%d | batch_size=%d | device=%s | devices=%d | seed=%d",
                args.mode, args.zero_shot, args.composition, args.checkpoint.resolve(),
                statistic_path.resolve() if statistic_path is not None else None, args.output.resolve(),
                args.num_samples, args.batch_size, ",".join(map(str, devices)), len(devices), args.seed,
            )
            with tqdm(total=args.num_samples, desc="Generating", unit="crystal", dynamic_ncols=True,
                      position=0, disable=not args.show_progress) as pbar:
                if len(devices) > 1:
                    progress = mp.get_context("spawn").Value("q", 0)
                    previous_lock = tqdm.get_lock()
                    tqdm.set_lock(progress.get_lock())
                    try:
                        workers = mp.spawn(
                            _generate_worker, args=(args, statistic_path, devices, progress),
                            nprocs=len(devices), join=False,
                        )
                        while True:
                            finished = workers.join(timeout=0.1)
                            pbar.update(progress.value - pbar.n)
                            if finished:
                                break
                    finally:
                        tqdm.set_lock(previous_lock)
                    saved = progress.value
                    if saved != args.num_samples:
                        raise RuntimeError(f"Generation stopped after saving {saved}/{args.num_samples} crystals")
                else:
                    saved = 0
                    for batch_count in _generate_cifs(args, statistic_path, devices[0], sample_indices, args.seed):
                        saved += batch_count
                        _report_progress(logger, args, saved)
                        pbar.update(batch_count)
            logger.info("Generation completed in %.2f s", perf_counter() - started)
        except (Exception, KeyboardInterrupt):
            logger.exception("Generation failed")
            raise


if __name__ == "__main__":
    main()
