import math
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from pcdet.utils import common_utils


class DeformableFusionModule(nn.Module):
    """
    query_feats:      (N, C)
    reference_points: (N, 2), normalized grid coords in [-1, 1]
    img_feats_map:    (B, C, H, W)
    batch_inds:       (N,)
    fov_mask:         (N,)
    """

    def __init__(self, embed_dim, num_heads=8, num_points=8,
                 dropout=0.1, max_offset_px=2.0):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        self.max_offset_px = max_offset_px

        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)

        # Zero-init: sampling starts at the projected reference point.
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)

        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.constant_(self.attention_weights.bias, 0.)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query_feats, reference_points, img_feats_map, batch_inds, fov_mask=None):
        N, C = query_feats.shape
        B, C_img, H, W = img_feats_map.shape
        assert C == self.embed_dim
        assert C_img == self.embed_dim

        # 1. Project values and split them into heads.
        # (B, C, H, W) -> (B, num_heads, head_dim, H, W)
        value = img_feats_map.permute(0, 2, 3, 1).contiguous()
        value = self.value_proj(value)
        value = value.permute(0, 3, 1, 2).contiguous()
        value = value.view(B, self.num_heads, self.head_dim, H, W)

        # 2. Per-head offsets, in feature-map pixels.
        raw_offsets = self.sampling_offsets(query_feats).view(
            N, self.num_heads, self.num_points, 2
        )
        offsets_px = self.max_offset_px * torch.tanh(raw_offsets)

        pixel_scale = query_feats.new_tensor(
            [2.0 / max(W, 1), 2.0 / max(H, 1)]
        ).view(1, 1, 1, 2)
        offsets = offsets_px * pixel_scale

        # 3. head-wise attention weights
        attn_logits = self.attention_weights(query_feats).view(
            N, self.num_heads, self.num_points
        )
        attn_weights = F.softmax(attn_logits, dim=-1)

        # 4. sampling locations
        ref = reference_points.view(N, 1, 1, 2)
        sampling_locations = ref + offsets  # (N, H, P, 2)

        output = query_feats.new_zeros(N, self.num_heads, self.head_dim)

        for b in range(B):
            mask = (batch_inds == b)
            if fov_mask is not None:
                mask = mask & fov_mask
            if not mask.any():
                continue

            cur_value = value[b]  # (num_heads, head_dim, H, W)
            cur_locs = sampling_locations[mask]  # (N_b, num_heads, num_points, 2)
            N_b = cur_locs.shape[0]

            # Treat each head as a batch so it samples its own channels.
            # grid: (num_heads, N_b * num_points, 1, 2)
            cur_grid = cur_locs.permute(1, 0, 2, 3).reshape(
                self.num_heads, N_b * self.num_points, 1, 2
            )

            sampled = F.grid_sample(
                cur_value,
                cur_grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )  # (num_heads, head_dim, N_b*num_points, 1)

            sampled = sampled.view(
                self.num_heads, self.head_dim, N_b, self.num_points
            ).permute(2, 0, 3, 1).contiguous()
            # (N_b, num_heads, num_points, head_dim)

            cur_w = attn_weights[mask].unsqueeze(-1)
            # (N_b, num_heads, num_points, 1)

            output[mask] = (sampled * cur_w).sum(dim=2)
            # (N_b, num_heads, head_dim)

        output = output.reshape(N, C)
        output = self.output_proj(output)
        output = self.dropout(output)
        output = self.norm(output)

        if fov_mask is not None:
            output = output * fov_mask.unsqueeze(-1).to(output.dtype)

        return output


class PAAF(nn.Module):
    """Sparse-to-dense fusion with hard projection, deformable sampling, and an adaptive gate."""

    def __init__(self,
                 voxel_channels,
                 mid_channels,
                 out_channels,
                 voxel_stride,
                 voxel_layer,
                 img_levels=[0, 1, 2, 3],
                 img_channels=256,
                 activate_out=True,
                 fuse_out=True,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=False,
                 lateral_conv=True,
                 reduction=16,
                 pc_range=None,
                 voxel_size=None,
                 num_heads=8,
                 num_points=8):
        super(PAAF, self).__init__()

        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)
        assert pc_range is not None and voxel_size is not None

        self.img_levels = img_levels
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.reduction = reduction
        self.out_channles = mid_channels
        self.voxel_stride = voxel_stride
        self.voxel_layer = voxel_layer

        self.register_buffer('point_cloud_range',
                             torch.as_tensor(pc_range, dtype=torch.float32).view(-1))
        self.register_buffer('voxel_size',
                             torch.as_tensor(voxel_size, dtype=torch.float32).view(-1))

        norm_1d_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # Voxel Feature Processor
        self.voxel_transform = nn.Sequential(
            nn.Linear(voxel_channels, mid_channels),
            norm_1d_layer(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Depth Embedding
        self.depth_emb = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Shared Layers (MLP for Hard Alignment features)
        self.shared_layers_lidar = nn.Sequential(
            nn.Linear(mid_channels, 256),
            nn.ReLU(),
            nn.Linear(256, mid_channels),
            norm_1d_layer(mid_channels),
        )

        self.shared_layers_image = nn.Sequential(
            nn.Linear(mid_channels, 256),
            nn.ReLU(),
            nn.Linear(256, mid_channels),
            nn.LayerNorm(mid_channels),
        )

        self.deformable_fusion = DeformableFusionModule(
            embed_dim=mid_channels,
            num_heads=num_heads,
            num_points=num_points,
        )

        # Adaptive Gated Fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(mid_channels * 2, mid_channels),
            nn.Sigmoid()
        )

        # Negative bias keeps the gate small so training starts from LiDAR features.
        nn.init.constant_(self.fusion_gate[0].bias, -2.0)

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                norm_1d_layer(out_channels),
                nn.ReLU(inplace=True))

    def forward(self, sp_voxel, batch_dict, indice):
        """Fuse one sparse voxel scale with the matching image feature map."""
        # 1. Image feature map for this voxel scale.
        batch_size = batch_dict['batch_size']
        if indice == 1:
            img_feats_map = batch_dict['img_fpn_feats_ori_shape'][self.voxel_layer - 1]
        else:
            img_feats_map = batch_dict['img_res_feats_ori_shape'][self.voxel_layer - 1]
        # img_feats_map shape: (B, C, H, W)

        # 2. Voxel centers in the LiDAR frame.
        batch_index = sp_voxel.indices[:, 0]
        spatial_zyx = sp_voxel.indices[:, 1:].to(dtype=self.voxel_size.dtype) * \
                      float(self.voxel_stride) + 0.5 * float(self.voxel_stride)
        z = spatial_zyx[:, 0] * self.voxel_size[2] + self.point_cloud_range[2]
        y = spatial_zyx[:, 1] * self.voxel_size[1] + self.point_cloud_range[1]
        x = spatial_zyx[:, 2] * self.voxel_size[0] + self.point_cloud_range[0]
        voxels_xyz = torch.stack((x, y, z), dim=1)
        calibs = batch_dict['calib']

        voxels_feats = self.voxel_transform(sp_voxel.features)

        # 3. Geometric projection and normalized reference points.
        img_feats_hard_aligned = []
        norm_reference_points_list = []
        fov_mask_list = []

        for b in range(batch_size):
            calib_batch = calibs[b]
            mask = (batch_index == b)
            voxels_3d_batch = voxels_xyz[mask]
            voxels_feats_batch = voxels_feats[mask]
            img_feat_map_batch = img_feats_map[b]  # (C, H, W)

            img_pts_batch, norm_ref_pts_b, fov_mask_b = self.point_sample_and_ref_gen(
                batch_dict, b, voxels_feats_batch, img_feat_map_batch,
                voxels_3d_batch, calib_batch
            )
            img_feats_hard_aligned.append(img_pts_batch)
            norm_reference_points_list.append(norm_ref_pts_b)
            fov_mask_list.append(fov_mask_b)

        img_feats_hard_aligned = torch.cat(img_feats_hard_aligned)         # (N, C)
        norm_reference_points = torch.cat(norm_reference_points_list)       # (N, 2)
        fov_mask = torch.cat(fov_mask_list)                                 # (N,) bool

        # 4. Shared LiDAR and image projections.
        shared_lidar = self.shared_layers_lidar(voxels_feats)
        shared_image_hard = self.shared_layers_image(img_feats_hard_aligned)

        fov_mask_f = fov_mask.unsqueeze(-1).to(shared_image_hard.dtype)
        shared_image_hard = shared_image_hard * fov_mask_f

        # 5. Deformable sampling. Out-of-FOV voxels skip grid_sample and stay zero.
        deformable_feat = self.deformable_fusion(
            query_feats=shared_lidar,
            reference_points=norm_reference_points,
            img_feats_map=img_feats_map,
            batch_inds=batch_index,
            fov_mask=fov_mask,
        )

        # 6. Adaptive gated fusion. LiDAR is the residual base.
        image_info = shared_image_hard + deformable_feat
        cat_feat = torch.cat([shared_lidar, image_info], dim=1)
        gate = self.fusion_gate(cat_feat)
        gate = gate * fov_mask_f
        fuse_out = shared_lidar + gate * image_info

        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        sp_voxel = sp_voxel.replace_feature(fuse_out)
        return sp_voxel

    def get_output_feature_dim(self):
        return self.out_channles

    def point_sample_and_ref_gen(self, batch_dict, batch_index,
                                 voxels_feats, img_feats_map, voxels_3d, calib):
        """
        Args:
            voxels_feats: (N, C_mid), features after voxel_transform.
            img_feats_map: (C_img, H_feat, W_feat), image feature map of this sample.
            voxels_3d: (N, 3), augmented LiDAR xyz seen by the network.
        Returns:
            image_pts: (N, C_img), hard-aligned image features, 0 outside the FOV.
            norm_ref_pts: (N, 2), grid_sample coords in [-1, 1], placeholder outside the FOV.
            fov_mask: (N,) bool, True when the projection falls inside the image.
        """
        h_feat, w_feat = img_feats_map.shape[1:]
        h_orig, w_orig = batch_dict['images'].shape[2:]

        stride_x = w_orig / w_feat
        stride_y = h_orig / h_feat

        voxels_3d_proj = voxels_3d.clone()

        if 'noise_scale' in batch_dict:
            voxels_3d_proj = voxels_3d_proj / batch_dict['noise_scale'][batch_index]
        if 'noise_rot' in batch_dict:
            voxels_3d_proj = common_utils.rotate_points_along_z(
                voxels_3d_proj.unsqueeze(0),
                -batch_dict['noise_rot'][batch_index].unsqueeze(0)
            )[0]
        # OpenPCDet flip_x negates y, and flip_y negates x.
        if 'flip_x' in batch_dict and bool(batch_dict['flip_x'][batch_index]):
            voxels_3d_proj = voxels_3d_proj.clone()
            voxels_3d_proj[:, 1] = -voxels_3d_proj[:, 1]
        if 'flip_y' in batch_dict and bool(batch_dict['flip_y'][batch_index]):
            voxels_3d_proj = voxels_3d_proj.clone()
            voxels_3d_proj[:, 0] = -voxels_3d_proj[:, 0]

        # Project onto the original image, then map pixels onto the feature map.
        voxels_2d, depth = calib.lidar_to_img(voxels_3d_proj.detach().cpu().numpy())
        voxels_2d_tensor = torch.as_tensor(
            voxels_2d, dtype=voxels_feats.dtype, device=voxels_feats.device
        )

        feat_x = voxels_2d_tensor[:, 0] / stride_x
        feat_y = voxels_2d_tensor[:, 1] / stride_y

        depth_tensor = torch.as_tensor(depth, dtype=voxels_feats.dtype, device=voxels_feats.device)
        # Positive depth and inside the feature map.
        fov_mask = (
            (depth_tensor > 1e-5) &
            (feat_x >= 0) & (feat_x <= w_feat - 1) &
            (feat_y >= 0) & (feat_y <= h_feat - 1)
        )

        # grid_sample normalization for align_corners=False.
        ref_pts_u = 2.0 * (feat_x + 0.5) / w_feat - 1.0
        ref_pts_v = 2.0 * (feat_y + 0.5) / h_feat - 1.0
        norm_ref_pts = torch.stack([ref_pts_u, ref_pts_v], dim=-1)
        # Points outside [-1, 1] make zero-padding return 0. fov_mask also skips them.
        out_of_fov_filler = norm_ref_pts.new_full((2,), -2.0)
        norm_ref_pts = torch.where(
            fov_mask.unsqueeze(-1), norm_ref_pts, out_of_fov_filler
        )

        # Hard-aligned features by integer pixel index. Outside the FOV stays 0.
        image_pts = torch.zeros(
            (voxels_feats.shape[0], img_feats_map.shape[0]),
            device=img_feats_map.device,
            dtype=img_feats_map.dtype,
        )
        if fov_mask.any():
            valid_u = feat_x[fov_mask].clamp(0, w_feat - 1).long()
            valid_v = feat_y[fov_mask].clamp(0, h_feat - 1).long()
            image_pts[fov_mask] = img_feats_map[:, valid_v, valid_u].permute(1, 0).to(
                image_pts.dtype
            )

        # Depth embedding uses the augmented xyz, matching the coordinates the network sees.
        depth_emb = self.depth_emb(voxels_3d)
        if self.training:
            image_pts = F.dropout(image_pts * depth_emb, p=self.dropout_ratio)
        else:
            image_pts = image_pts * depth_emb

        return image_pts, norm_ref_pts, fov_mask
