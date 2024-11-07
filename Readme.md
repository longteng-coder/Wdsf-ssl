# Wds-ssf
This is a implementation of the paper “A Weld Defect Segmentation Framework Based on Visual Self-Supervised Learning”


* This repo is based on [`MAE in PyTorch+GPU`](https://github.com/facebookresearch/mae). Installation and preparation follow that repo.


* Radiographic inspection of weld defect dataset(RTWD) is constructed by a semi-automatic data extraction engine. Detailed explanation in [RTWD.md](RTWD/RTWD.md).


* Instructions for using object level masking strategy:
```python
from obj_mask import create_objmask, modify_mask

class MaskedAutoencoderViT(nn.Module):
    ...
    def random_masking(self, x, lt, mask_ratio, obj_mask_ratio):
        """
        lt: Coordinates of the object area.
        mask_ratio: The masking ratio of the entire image inherited from MAE.
        obj_mask_ratio: Masking ratio of object area.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # Sample within the object area and modify noise.
        object_mask_index = create_objmask(self.grid_size, self.patch_size, lt, obj_mask_ratio)
        noise = modify_mask(noise, object_mask_index)
```
* Implementation of multi-head decoding segmentation is in [ssf_seg.py](ssf_seg.py).