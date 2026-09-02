"""Entry point for the two-stage training pipeline (and checkpoint evaluation).

Stage 1 (model.stage=1, data=stage1_dense): jointly train the shared BRDF
decoder and per-point latent/normal/tangent tables on the multi-material
observation tensors (the "material prior" of the paper).

Stage 2 (model.stage=2, data=stage2_dense): freeze the decoder (loaded
decoder-only from model.ckpt_path) and fit a dense 2048^2 latent texture,
neural-geometry UV offset and per-channel scale beta for one material.

Evaluation (model.test=True): load ALL non-emitter weights from a trained
stage-2 checkpoint and run the exact validation step over the held-out
views — this is how the paper's PSNR tables are produced (scripts/eval_*).

Trainer dispatch: get_trainer_class(stage, data.dataset_name); RoboCloth data
uses Stage1Trainer/Stage2Trainer, the comparison datasets (bonn/merl/ubo) use
their thin adapter subclasses. Config composition is Hydra (configs/).
"""
import torch
import torch.nn as nn
import torchvision
import json
import os
from tqdm import tqdm
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from renderer import ForwardRenderer
from trainers import get_trainer_class
from models.brdf import SvPBRBRDF
from utils.checkpoint_io import load_checkpoint_file, extract_state_dict, load_state_dict_strict
from torch.utils.data import DataLoader
from datasets import (
    RealImageDenseDataset, RealValDataset, MultiMaterialDenseDataset,
    MERLBRDFIterableDataset, MERLBRDFFixedDataset,
    BonnDataset, BonnValDataset, BonnSingleMaterialDataset, BonnSingleMaterialValDataset,
    UBOBTFTrainDataset, UBOBTFValDataset,
)
import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging
import cv2
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Explicit partial-load allowlists (utils/checkpoint_io.load_state_dict_strict).
# Every key inside an allowlist must be present with a matching shape; keys
# outside it are reported as intentionally skipped. Nothing else is tolerated.
#   Stage-2 warm start / validate_on_stage1: exactly the shared BRDF decoder
#   (the paper's decoder-only transfer; cross-arch warm starts work as long
#   as the decoder MLP shape matches).
DECODER_ONLY_PREFIXES = ('material.decoder.',)
#   Stage-1 warm start: the whole material (decoder, latent bank, offsets).
MATERIAL_PREFIXES = ('material.',)
#   Evaluation: every weight, except the emitter buffers, which are rebuilt
#   from the calibration files at construction time (shapes follow the data).
EVAL_IGNORED_PREFIXES = ('emitter.',)


def restore_evaluation_texture_resolution(cfg, checkpoint):
    """Use the stage-2 texture resolution recorded in a checkpoint.

    Lightning stores the composed Hydra config under ``hyper_parameters``.
    Evaluation constructs the material before loading its tensors, so using
    the current config's default resolution can otherwise create a latent
    texture with the wrong shape.  Training remains controlled by the current
    experiment config; only checkpoint evaluation restores this architectural
    value automatically.
    """
    if not bool(cfg.model.get('test', False)) or int(cfg.model.get('stage', 1)) != 2:
        return None

    hparams = checkpoint.get('hyper_parameters') or {}
    material_hparams = hparams.get('material', {}) if hasattr(hparams, 'get') else {}
    saved_resolution = (
        material_hparams.get('texture_resolution')
        if hasattr(material_hparams, 'get') else None
    )
    if saved_resolution is None:
        return None

    saved_resolution = int(saved_resolution)
    current_resolution = cfg.material.get('texture_resolution', None)
    if current_resolution is None or int(current_resolution) != saved_resolution:
        with open_dict(cfg.material):
            cfg.material.texture_resolution = saved_resolution
        print(
            "[checkpoint config] restored material.texture_resolution="
            f"{saved_resolution} for evaluation"
        )
    return saved_resolution

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)

    # Fail closed on the checkpoint path itself: a set-but-missing file, or an
    # evaluation without any checkpoint, must stop here instead of silently
    # training / validating a randomly initialised model.
    ckpt_path = cfg.model.ckpt_path or None
    is_test = bool(cfg.model.get('test', False))
    if is_test and ckpt_path is None:
        raise ValueError("model.test=True requires model.ckpt_path (the checkpoint to evaluate)")
    if ckpt_path is not None and not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"model.ckpt_path does not exist: {ckpt_path}")

    # A stage-2 checkpoint's latent-texture shape is architecture-critical.
    # Read it before constructing the material, then reuse the same checkpoint
    # object below so evaluation does not load a multi-GB file twice.
    evaluation_checkpoint = None
    if is_test and int(cfg.model.get('stage', 1)) == 2:
        evaluation_checkpoint = load_checkpoint_file(ckpt_path, map_location='cpu')
        restore_evaluation_texture_resolution(cfg, evaluation_checkpoint)

    print(f"[DIAG] About to create output dir: {cfg.exp_output_root_path}")
    os.makedirs(cfg.exp_output_root_path, exist_ok=True)
    print("[DIAG] Output dir created.")
    checkpoint_output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(checkpoint_output_path, exist_ok=True)

    # Load ground truth material parameters from pbr config
    print("[DIAG] Before hydra.compose for gt_material_cfg...")
    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=svpbr"]).material
    print("[DIAG] After hydra.compose for gt_material_cfg.")
    
    # Use ground truth parameters from pbr.yaml
    albedo = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic = gt_material_cfg.metallic
    
    output_folder = os.path.join(cfg.exp_output_root_path, f'fabric_pattern_07_4k')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize material dynamically from module.type config
    print("[DIAG] Before material class import...")
    material_module = importlib.import_module(cfg.material.module)
    material_class = getattr(material_module, cfg.material.type)
    print(f"[DIAG] Before material_class({cfg.material.type}) constructor...")
    material = material_class(cfg.material)
    print("[DIAG] After material constructor.")
    gt_material = SvPBRBRDF(
        cfg=gt_material_cfg,
        albedo=torch.tensor(albedo)
    )  # Ground truth uses pbr config
    print("before trainer init")
    # Get the appropriate trainer class based on stage and data type
    stage = cfg.model.get('stage', 1)  # Default to stage 1
    data_type = cfg.data.get('dataset_name', 'default')
    TrainerClass = get_trainer_class(stage, data_type)
    
    print(f"Using trainer for stage {stage}: {TrainerClass.__name__}")
    model = TrainerClass(cfg, material, gt_material, roughness, metallic)
    
    # Track whether to use Lightning's resume functionality
    resume_ckpt_path = None
    
    if ckpt_path is not None:
        print(f"=> loading model checkpoint '{ckpt_path}'")

        # If continue_training is enabled, use PyTorch Lightning's native resume
        # This will restore model weights, optimizer state, scheduler state, epoch, etc.
        continue_training = getattr(cfg.model, 'continue_training', False)
        if continue_training:
            print(f"=> continue_training=True: will resume full training state from checkpoint")
            resume_ckpt_path = ckpt_path
        else:
            # Manual weight loading for transfer learning / partial loading.
            # Every branch goes through load_state_dict_strict: a missing,
            # unexpected or shape-mismatched tensor raises, and partial loads
            # are limited to the explicit allowlists at the top of this file.
            checkpoint = evaluation_checkpoint
            if checkpoint is None:
                checkpoint = load_checkpoint_file(ckpt_path, map_location='cpu')
            state_dict = extract_state_dict(checkpoint, source=ckpt_path)

            if stage == 2:
                if cfg.model.test:
                    # Evaluation: every weight comes from the checkpoint; only
                    # the emitter buffers are skipped (rebuilt at construction).
                    report = load_state_dict_strict(
                        model, state_dict, ignore_prefixes=EVAL_IGNORED_PREFIXES, source=ckpt_path)
                    print(f"=> loaded model checkpoint successfully (excluding emitter). {len(report.loaded)}/{len(state_dict)} parameters loaded.")
                    print(report.summary())
                else:
                    # Stage 2: decoder-only warm start — load exactly the
                    # material.decoder.* tensors (all of them, shapes checked);
                    # latent texture / neural geometry / factor start fresh.
                    report = load_state_dict_strict(
                        model, state_dict, allow_prefixes=DECODER_ONLY_PREFIXES, source=ckpt_path)
                    print(f"=> Stage 2: loaded decoder checkpoint successfully. {len(report.loaded)} decoder parameters loaded.")
                    print(report.summary())

                    # Optionally override learnable_factor init (e.g. compensate for a
                    # decoder whose output range is far from the GT range).
                    factor_init = getattr(cfg.model, 'factor_init', None)
                    if factor_init is not None and getattr(model.material, 'learnable_factor', False):
                        with torch.no_grad():
                            model.material.factor.data.fill_(float(factor_init))
                        print(f"=> Stage 2: initialized learnable_factor to {float(factor_init)}")

                    # If use_latent_bank is enabled, also load the latent bank from checkpoint
                    use_latent_bank = getattr(cfg.material, 'use_latent_bank', False)
                    if use_latent_bank:
                        latent_bank_key = 'material.point_latent_bank.weight'
                        if latent_bank_key not in state_dict:
                            # The model cannot run without a bank: fail here, not
                            # later with an opaque error from a None module.
                            raise RuntimeError(
                                f"material.use_latent_bank=True but {ckpt_path} has no {latent_bank_key}")
                        latent_weights = state_dict[latent_bank_key]
                        num_points, latent_dim = latent_weights.shape
                        # Create embedding from checkpoint weights directly
                        model.material.point_latent_bank = nn.Embedding(num_points, latent_dim)
                        model.material.point_latent_bank.weight.data = latent_weights
                        print(f"=> Stage 2: loaded latent bank from checkpoint: {num_points} x {latent_dim}")

                    # If initialize_from_std is enabled, reinitialize latent texture using std from checkpoint's latent bank
                    initialize_from_std = getattr(cfg.material, 'initialize_from_std', False)
                    if initialize_from_std:
                        latent_bank_key = 'material.point_latent_bank.weight'
                        if latent_bank_key in state_dict:
                            latent_weights = state_dict[latent_bank_key]
                            # latent_weights: [num_points, latent_dim]
                            # Last 6 dimensions have special meaning (normal + tangent), exclude them
                            brdf_latent_weights = latent_weights[:, :-6]
                            
                            # Compute std from the brdf latent dimensions
                            computed_std = brdf_latent_weights.std().item()
                            
                            # Reinitialize only the first N-6 dimensions, keep last 6 unchanged
                            latent_texture = model.material.latent_texture
                            resolution = latent_texture.resolution
                            num_brdf_dims = brdf_latent_weights.shape[1]
                            
                            # Reinitialize first N-6 dimensions with computed std
                            latent_texture.params.data[:, :num_brdf_dims, :, :] = torch.randn(
                                1, num_brdf_dims, resolution, resolution,
                                device=latent_texture.params.device,
                                dtype=latent_texture.params.dtype
                            ) * computed_std
                            
                            print(f"=> Stage 2: initialized latent texture (first {num_brdf_dims} dims) from checkpoint std={computed_std:.6f}")
                        else:
                            print(f"=> Stage 2: initialize_from_std=True but no latent bank weights found in checkpoint.")
            else:
                # Stage 1
                validate_on_stage1 = getattr(cfg.model, 'validate_on_stage1', False)
                if validate_on_stage1 and not cfg.model.test:
                    # Validate-on-stage1: only load decoder weights so latents
                    # are optimised from scratch on the val split (mirrors
                    # stage 2's non-test branch). Latent bank stays at the
                    # current model's fresh initialisation.
                    report = load_state_dict_strict(
                        model, state_dict, allow_prefixes=DECODER_ONLY_PREFIXES, source=ckpt_path)
                    print(f"=> Stage 1 (validate_on_stage1): loaded decoder checkpoint successfully. {len(report.loaded)} decoder parameters loaded.")
                    print(report.summary())
                else:
                    # Stage 1: load the whole material sub-tree (decoder, latent
                    # bank, offsets) — all of it; trainer-side state stays fresh.
                    state_dict = dict(state_dict)

                    # Debug: inject latents for one hardcoded material only (offset-aware)
                    debug_load_material_id = 1
                    latent_bank_key = 'material.point_latent_bank.weight'
                    if cfg.data.debug and latent_bank_key in state_dict:
                        ckpt_latents = state_dict.pop(latent_bank_key)   # [N_ckpt, D]
                        if hasattr(model.material, 'material_offset_tensor'):
                            offset = model.material.material_offset_tensor[debug_load_material_id].item()
                        else:
                            offset = 0  # single-material mode: no global offset
                        n = ckpt_latents.shape[0]
                        # state_dict() tensors share storage with the parameters,
                        # so this writes the rows into the live bank in place;
                        # the bank then re-enters the dict so the strict scope
                        # check sees it present (loading it onto itself is a no-op).
                        bank = model.state_dict()[latent_bank_key]
                        bank[offset:offset + n] = ckpt_latents
                        state_dict[latent_bank_key] = bank
                        print(f"[Debug] Injected latents for material {debug_load_material_id}: "
                              f"ckpt rows 0:{n} → bank rows {offset}:{offset + n}")

                    report = load_state_dict_strict(
                        model, state_dict, allow_prefixes=MATERIAL_PREFIXES, source=ckpt_path)
                    print(f"=> loaded material checkpoint successfully. {len(report.loaded)} material parameters loaded.")
                    print(report.summary())
    print("after trainer init")
    print("==> initializing data ...")
    validate_on_stage1 = getattr(cfg.model, 'validate_on_stage1', False) and (cfg.model.stage == 1) and (not cfg.model.test)
    if cfg.data.dataset_name == "stage2_dense":
        if not cfg.model.test:
            train_dataset = RealImageDenseDataset(cfg, gt_folder=cfg.gt_folder, split="train")
        val_dataset = RealValDataset(cfg, gt_folder=cfg.gt_folder)
    elif cfg.data.dataset_name == "merl":
        if cfg.model.stage == 1:
            train_dataset = MERLBRDFIterableDataset(cfg,data_folder=cfg.dataset_folder,batch_size=cfg.data.rays_num,split="train")
            if cfg.data.debug & cfg.data.valid_on_train_set:
                val_dataset = MERLBRDFIterableDataset(cfg,data_folder=cfg.dataset_folder,batch_size=cfg.data.rays_num,split="val")
            else:
                val_dataset = MERLBRDFIterableDataset(cfg,data_folder=cfg.dataset_folder,batch_size=cfg.data.rays_num,split="val")
        else:
            train_dataset = MERLBRDFFixedDataset(cfg,data_folder=cfg.dataset_folder,batch_size=100,split="train")
            if cfg.data.debug & cfg.data.valid_on_train_set:
                val_dataset = MERLBRDFFixedDataset(cfg,data_folder=cfg.dataset_folder,batch_size=1048576,split="val")
            else:
                val_dataset = MERLBRDFFixedDataset(cfg,data_folder=cfg.dataset_folder,batch_size=1048576,split="val")
    elif cfg.data.dataset_name == "stage1_dense":
        if validate_on_stage1:
            # Train on the held-out test materials using the iterable train
            # loader (mirrors the Bonn flow which points dataset_folder at
            # Bonn_val). Caller sets data.training_list_path to e.g.
            # test_list_420.txt so MultiMaterialLatentBRDF and the dataloader
            # both see the test materials.
            train_dataset = MultiMaterialDenseDataset(cfg, root_folder=cfg.dataset_folder, split="train")
            val_dataset = None
        else:
            train_dataset = MultiMaterialDenseDataset(cfg, root_folder=cfg.dataset_folder, split="train")
            if cfg.data.debug & cfg.data.valid_on_train_set:
                val_dataset = MultiMaterialDenseDataset(cfg, root_folder=cfg.dataset_folder, split="train", share_from=train_dataset)
            else:
                val_dataset = MultiMaterialDenseDataset(cfg, root_folder=cfg.dataset_folder, split="val", share_from=train_dataset)
    elif cfg.data.dataset_name == "bonn":
        if cfg.model.stage == 1:
            if cfg.model.test:
                val_dataset = BonnValDataset(cfg, root_folder=cfg.dataset_folder)
            elif validate_on_stage1:
                # Train on the val-set materials (cfg.dataset_folder pointed at
                # Bonn_val) using the iterable train loader. Skip BonnValDataset
                # because it hardcodes mat_id=1, which only exists in Bonn_train.
                train_dataset = BonnDataset(cfg, root_folder=cfg.dataset_folder, split="train")
                val_dataset = None
            else:
                train_dataset = BonnDataset(cfg, root_folder=cfg.dataset_folder, split="train")
                val_dataset = BonnValDataset(cfg, root_folder=cfg.dataset_folder)

        else:
            train_dataset = BonnSingleMaterialDataset(cfg, root_folder=cfg.dataset_folder, split="train")
            if cfg.data.debug & cfg.data.valid_on_train_set:
                val_dataset = BonnSingleMaterialValDataset(cfg, root_folder=cfg.dataset_folder)
            else:
                val_dataset = BonnSingleMaterialValDataset(cfg, root_folder=cfg.dataset_folder)

    elif cfg.data.dataset_name == "ubo":
        btf_path = os.path.join(cfg.dataset_folder, cfg.data.btf_filename)
        train_dataset = UBOBTFTrainDataset(cfg, btf_path=btf_path, split='train')
        val_dataset = UBOBTFValDataset(cfg, btf_path=btf_path)

    else:
        raise ValueError(f"Invalid dataset name: {cfg.data.dataset_name}")
    if not cfg.model.test:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=False,
        )
    if val_dataset is None:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=cfg.data.num_workers,
        )
    print("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.model.logger, save_dir=cfg.exp_output_root_path)

    print("==> initializing monitor ...")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    enable_ckpt = cfg.model.trainer.get('enable_checkpointing', True)
    if enable_ckpt:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cfg.model.checkpoint_monitor.dirpath, f'model_{roughness:.2f}_{metallic:.2f}'),
            filename=cfg.model.checkpoint_monitor.filename,
            save_top_k=cfg.model.checkpoint_monitor.save_top_k,
            every_n_epochs=cfg.model.checkpoint_monitor.every_n_epochs,
            save_on_train_epoch_end=cfg.model.checkpoint_monitor.save_on_train_epoch_end,
            save_last=cfg.model.checkpoint_monitor.save_last,
        )
        callbacks = [checkpoint_callback, lr_monitor]
    else:
        callbacks = [lr_monitor]

    print("==> initializing trainer ...")

    # PL 1.x doesn't accept the `ddp_find_unused_parameters_{true,false}` string
    # aliases (added in PL 2.x), so map them to a DDPStrategy instance.
    trainer_kwargs = dict(cfg.model.trainer)
    _strategy = trainer_kwargs.get('strategy')
    if _strategy == 'ddp_find_unused_parameters_true':
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=True)
    elif _strategy == 'ddp_find_unused_parameters_false':
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        callbacks=callbacks, logger=logger,
        # track_grad_norm=2,  # Disabled: broken with automatic_optimization=False
        # gradient_clip_val=1.0,  # Optional: clip gradients
        **trainer_kwargs
    )
    # tracer = VizTracer()
    # tracer.start()
    if cfg.model.test:
        trainer.validate(model, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt_path)
    # tracer.stop()
    # tracer.save(f"Ray-rect-intersection_tracer.json")
    """  Skipping testing for now """
    # test_results = trainer.test(model, dataloaders=test_loader)

    # test_psnr = sum(result['test/psnr'] for result in test_results) / len(test_results)
    # print(f"PSNR for roughness {roughness:.2f}, metallic {metallic:.2f}: {test_psnr:.2f}")

    torch.cuda.empty_cache()

    print('Training and Testing Complete!')


if __name__ == "__main__":
    main()
