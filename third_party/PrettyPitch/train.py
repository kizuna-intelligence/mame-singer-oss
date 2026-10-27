"""PrettyPitch の F0 モデルの組み立て（infer.load_model が使う部分だけ）。"""
from prettypitch.models import F0Diffusion3ch, F0Reflow3ch


def build_model(ckpt_cfg: dict):
    common = dict(
        n_phonemes=int(ckpt_cfg["n_phonemes"]),
        n_speakers=int(ckpt_cfg["n_speakers"]),
        spk_dim=int(ckpt_cfg["spk_dim"]),
        n_styles=int(ckpt_cfg["n_styles"]),
        conditioner=str(ckpt_cfg.get("conditioner", "convnext")),
        convnext_dilations=ckpt_cfg.get("convnext_dilations"),
    )
    if str(ckpt_cfg.get("arch", "f0_3ch_diffusion")) == "f0_3ch_reflow":
        return F0Reflow3ch(num_steps=int(ckpt_cfg.get("infer_steps", 1)), **common)
    return F0Diffusion3ch(n_steps=int(ckpt_cfg["n_steps"]), **common)
