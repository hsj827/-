from __future__ import annotations

import numpy as np

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone:
    """Frozen ImageNet CNN used only for feature extraction - nothing is trained here."""

    def __init__(self, name: str = "resnet18", layers: tuple[str, ...] = ("layer2", "layer3"),
                 pretrained: bool = True, device: str | None = None):
        import torch
        import torchvision

        self.torch = torch
        self.name = name
        self.layers = tuple(layers)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        weights = "DEFAULT" if pretrained else None
        try:
            model = getattr(torchvision.models, name)(weights=weights)
        except Exception as exc:  # offline / no cached weights
            if not pretrained:
                raise
            raise RuntimeError(
                f"could not fetch pretrained weights for {name} ({exc}); "
                "set detectors.*.pretrained: false to run untrained, or pre-download the weights"
            ) from exc
        model.eval()
        model.to(self.device)
        self.model = model
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}

    def activations(self, image: np.ndarray, grid: tuple[int, int] | None = None) -> list[np.ndarray]:
        """Layer feature maps as (h, w, c). Resampling happens in torch - cv2.resize
        refuses the high-channel matrices these networks produce."""
        torch = self.torch
        tensor = torch.from_numpy(np.stack([image] * 3, axis=0))[None].to(self.device)
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        captured: list[np.ndarray] = []
        handles = []
        for layer_name in self.layers:
            module = dict(self.model.named_children())[layer_name]

            def _hook(_m, _in, out, sink=captured):
                sink.append(out.detach())

            handles.append(module.register_forward_hook(_hook))
        with torch.no_grad():
            self.model(((tensor - mean) / std))
        for handle in handles:
            handle.remove()

        maps = []
        for out in captured:
            features = out[0]
            if grid is not None and tuple(features.shape[1:]) != tuple(grid):
                mode = "area" if (features.shape[1] > grid[0] and features.shape[2] > grid[1]) else "bilinear"
                kwargs = {} if mode == "area" else {"align_corners": False}
                features = torch.nn.functional.interpolate(
                    features[None].float(), size=tuple(grid), mode=mode, **kwargs)[0]
            maps.append(np.ascontiguousarray(features.permute(1, 2, 0).cpu().numpy()))
        return maps

    def projection(self, channels: int, dim: int, seed: int) -> np.ndarray:
        key = (channels, dim, seed)
        if key not in self._cache:
            rng = np.random.default_rng(seed)
            self._cache[key] = (rng.standard_normal((channels, dim)) / np.sqrt(dim)).astype(np.float32)
        return self._cache[key]

    def patches(self, image: np.ndarray, grid: tuple[int, int], pool: int = 3,
                proj_dim: int | None = None, seed: int = 0) -> np.ndarray:
        """(h*w, d) L2-normalised patch descriptors on a fixed grid."""
        maps = self.activations(image, grid)
        feats = np.concatenate(maps, axis=-1).astype(np.float32)
        if pool and pool > 1:
            feats = _avg_pool_last_axis(feats, int(pool))
        if proj_dim:
            proj = self.projection(feats.shape[-1], proj_dim, seed)
            feats = feats @ proj
        flat = feats.reshape(-1, feats.shape[-1])
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        return (flat / np.maximum(norms, 1e-9)).astype(np.float32)


def _avg_pool_last_axis(feats: np.ndarray, pool: int) -> np.ndarray:
    h, w, _ = feats.shape
    pad = pool // 2
    padded = np.pad(feats, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(feats)
    for dy in range(pool):
        for dx in range(pool):
            out += padded[dy:dy + h, dx:dx + w]
    return out / float(pool * pool)
