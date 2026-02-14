from indextts_mlx.models.s2mel import MLXS2MelPipeline, create_mlx_s2mel_pipeline


def load_s2mel_pipeline(checkpoint_path: str = None) -> MLXS2MelPipeline:
    return create_mlx_s2mel_pipeline(checkpoint_path=checkpoint_path)
