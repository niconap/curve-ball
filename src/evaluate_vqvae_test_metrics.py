import os
import sys
import warnings
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from metrics.abstract_metrics import TrainAbstractMetrics, TrainAbstractMetricsDiscrete
from src.HGCN import HGCN
from src.HGVAE import HGVAE

warnings.filterwarnings("ignore", category=PossibleUserWarning)

_src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_src_dir, "models", "hyperbolic_nn_plusplus"))


def build_graph_eval_objects(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    if dataset_name not in ["sbm", "comm20", "planar", "tree", "ego_small"] and not dataset_name.startswith('synthetic_'):
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported by this evaluator. "
            "Use one of: sbm, comm20, planar, tree, ego_small."
        )

    from analysis.spectre_utils import (
        Comm20SamplingMetrics,
        EgoSmallSamplingMetrics,
        PlanarSamplingMetrics,
        SBMSamplingMetrics,
        TreeSamplingMetrics,
    )
    from analysis.visualization import NonMolecularVisualization
    from datasets.spectre_dataset import SpectreDatasetInfos, SpectreGraphDataModule

    datamodule = SpectreGraphDataModule(cfg)
    if dataset_name == "sbm":
        sampling_metrics = SBMSamplingMetrics(datamodule)
    elif dataset_name == "comm20":
        sampling_metrics = Comm20SamplingMetrics(datamodule)
    elif dataset_name == "tree":
        sampling_metrics = TreeSamplingMetrics(datamodule)
    elif dataset_name == "ego_small" or dataset_name.startswith('synthetic_'):
        sampling_metrics = EgoSmallSamplingMetrics(datamodule)
    else:
        sampling_metrics = PlanarSamplingMetrics(datamodule)

    dataset_infos = SpectreDatasetInfos(datamodule, cfg.dataset)
    train_metrics = (
        TrainAbstractMetricsDiscrete() if cfg.model.type == "HGVAE" else TrainAbstractMetrics()
    )
    visualization_tools = NonMolecularVisualization()

    if cfg.model.type == "HGVAE" and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
    }
    return datamodule, sampling_metrics, model_kwargs


def build_model(cfg: DictConfig, model_kwargs):
    if not cfg.model.use_poincare:
        cfg.model.lgcn_in_channels = model_kwargs["dataset_infos"].input_dims["X"] + 1
        cfg.model.lgcn_in_edge_channels = model_kwargs["dataset_infos"].input_dims["E"] + 1
        cfg.model.lgcn_out_channels = cfg.model.latent_channels
    else:
        cfg.model.lgcn_in_channels = (
            model_kwargs["dataset_infos"].input_dims["X"] + model_kwargs["dataset_infos"].input_dims["y"]
        )
        cfg.model.lgcn_in_edge_channels = model_kwargs["dataset_infos"].input_dims["E"]
        cfg.model.lgcn_out_channels = cfg.model.latent_channels

    cfg.model.edge_classes = model_kwargs["dataset_infos"].output_dims["E"]
    cfg.model.node_classes = model_kwargs["dataset_infos"].output_dims["X"]

    if cfg.train.hyper_model == "HGCN":
        model = HGCN(cfg, **model_kwargs)
    elif cfg.train.hyper_model == "HGVAE":
        model = HGVAE(cfg, **model_kwargs)
    else:
        raise ValueError(f"Unknown hyper model: {cfg.train.hyper_model}")
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> Tuple[int, int]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state_dict, strict=False)
    return len(incompatible.missing_keys), len(incompatible.unexpected_keys)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.train.seed)
    ckpt_path = cfg.general.test_only or cfg.general.resume
    if ckpt_path is None:
        raise ValueError(
            "No checkpoint provided. Set general.test_only=<path_to_checkpoint> "
            "or general.resume=<path_to_checkpoint>."
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    _, sampling_metrics, model_kwargs = build_graph_eval_objects(cfg)
    model = build_model(cfg, model_kwargs)
    missing_count, unexpected_count = load_checkpoint(model, ckpt_path)

    if not hasattr(model, "sample_batch"):
        raise AttributeError(
            f"Model {type(model).__name__} does not expose sample_batch(), required for sampling metrics."
        )

    device = torch.device("cuda:0" if cfg.general.gpus > 0 and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.print = print

    total_samples = int(cfg.general.final_model_samples_to_generate)
    if total_samples <= 0:
        raise ValueError("general.final_model_samples_to_generate must be > 0")
    sample_batch_size = int(max(1, 2 * cfg.train.batch_size))

    samples = []
    generated = 0
    with torch.no_grad():
        while generated < total_samples:
            to_generate = min(sample_batch_size, total_samples - generated)
            samples.extend(model.sample_batch(batch_size=to_generate, num_nodes=None, batch_id=generated))
            generated += to_generate

    test_metrics = sampling_metrics(
        samples,
        cfg.general.name,
        current_epoch=0,
        val_counter=-1,
        test=True,
        local_rank=0,
    )
    if test_metrics is None:
        raise RuntimeError("Sampling metrics did not return test metrics.")

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"State dict compatibility -- missing keys: {missing_count}, unexpected keys: {unexpected_count}")
    for key in ["degree_test", "clustering_test", "orbit_test"]:
        if key in test_metrics:
            print(f"{key}: {test_metrics[key]}")
        else:
            print(f"{key}: not available")


if __name__ == "__main__":
    main()