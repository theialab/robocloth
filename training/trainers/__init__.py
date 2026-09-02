from .stage1_trainer import Stage1Trainer
from .stage2_trainer import Stage2Trainer
from .stage1_trainer_merl import Stage1Trainer_MERL
from .stage2_trainer_merl import Stage2Trainer_MERL
from .stage1_trainer_bonn import Stage1Trainer_Bonn
from .stage2_trainer_bonn import Stage2Trainer_Bonn
from .stage2_trainer_ubo import Stage2Trainer_UBO

# Registry for easy lookup
TRAINER_REGISTRY = {
    'default': {
        1: Stage1Trainer,
        2: Stage2Trainer,
    },
    'merl': {
        1: Stage1Trainer_MERL,
        2: Stage2Trainer_MERL,
    },
    'bonn': {
        1: Stage1Trainer_Bonn,
        # Intentional: the Bonn implementation uses one trainer for both
        # phases.  For stage 2, train.py supplies BonnSingleMaterialDataset
        # and stage2_bonn.yaml freezes the shared decoder.  The older
        # Stage2Trainer_Bonn class is retained only for experiment history.
        2: Stage1Trainer_Bonn,
    },
    'ubo': {
        2: Stage2Trainer_UBO,
    },
}

def get_trainer_class(stage: int, data_type: str = 'default'):
    """Factory function to get trainer class by stage and data type.
    
    Args:
        stage: Training stage (1 or 2)
        data_type: Data config name ('merl' for MERL dataset, 'bonn' for Bonn, otherwise 'default')
    """
    registry_key = data_type if data_type in TRAINER_REGISTRY else 'default'
    registry = TRAINER_REGISTRY[registry_key]
    
    if stage not in registry:
        raise ValueError(f"Unknown stage: {stage}. Available: {list(registry.keys())}")
    return registry[stage]
