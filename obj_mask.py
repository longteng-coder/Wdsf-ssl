# Copyright (c) Xi'an Jiaotong University, School of Mechanical Engineering.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import torch


def generate_mask(num_rows, num_cols, counts,device):

    assert len(counts) == num_rows, "The length of zero_counts must match the number of rows."

    row_indices = torch.arange(num_cols).expand(num_rows, num_cols).to(device)
    mask = row_indices >= counts.view(-1, 1)

    return mask


def modify_mask(original_array, index_array):
    """
       Modify the token index list for MAE masking.
    """
    assert original_array.shape == index_array.shape, "Shapes of original and index arrays must match."

    num_rows, num_cols = original_array.shape

    for row in range(num_rows):
        indices = index_array[row]
        valid_indices = indices[indices != -1]

        original_array[row, valid_indices] = 1e5

    return original_array


def create_objmask(grid_size,patch_size,target_box,object_mask_ratio,device):
    """
    Sampling within the object area for masking.
    Target boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
           grid_size: Number of patches in the H and W directions of the image
           patch_size: Patch size
           target_box: Corner coordinates of the object(Tensor[M, 4])
           object_mask_ratio: Masking ratio of object
    Returns:
            object_mask_index:Tensor[N, num_patches]
    """
    w, h = grid_size
    y, x = torch.meshgrid([
        torch.arange(h),
        torch.arange(w)
    ])

    xy = torch.stack([x, y], dim=-1).float().view(-1, 2)
    # [H, W, 2] -> [HW, 2]

    grid_box = torch.cat((xy*patch_size, (xy+1)*patch_size), dim=-1).to(device)
    tbox = target_box.T
    area = (tbox[2] - tbox[0]) * (tbox[3] - tbox[1])
    inter = (torch.min(grid_box[:, None, 2:], target_box[:, 2:]) - torch.max(grid_box[:, None, :2], target_box[:, :2])).clamp(0).prod(2)
    inter_patch_num = torch.sum((inter > 0), dim=0)

    inter_norm = inter / area

    samples = torch.multinomial(inter_norm.T, num_samples=w*h, replacement=False)

    object_mask_num = (inter_patch_num * object_mask_ratio).int()
    object_mask = generate_mask(samples.size(0),samples.size(1),object_mask_num,device)
    object_mask_index = torch.where(object_mask, -1, samples)

    return object_mask_index


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    c = create_objmask((4,4),8,torch.tensor([[4.,5.,21.,18.],
                                      [3.,6.,16.,19.]]),0.5,device)

    d = torch.rand(2,16)
    e = modify_mask(d,c)
    print(c)
    print(e)