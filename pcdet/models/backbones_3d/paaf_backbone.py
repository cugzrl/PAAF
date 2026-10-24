from functools import partial
import spconv.pytorch as spconv
import time
import torch
from torch import nn
from ..backbones_2d.fuser.paaf import PAAF 
from ...utils.spconv_utils import replace_feature
from .spconv_backbone import post_act_block

class PAAFBackbone(nn.Module):
    @staticmethod
    def _sparse_add_by_coords(base_sp: spconv.SparseConvTensor, add_sp: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Add features of add_sp onto base_sp at matching coordinates.

        Indices need not share an order. The output keeps base_sp.indices so
        downstream sparse shapes stay stable. Only the coordinate intersection is updated.
        """
        if add_sp is None:
            return base_sp
        if base_sp.indices.numel() == 0 or add_sp.indices.numel() == 0:
            return base_sp

        # Fuse only when spatial shape and batch size match.
        if (list(base_sp.spatial_shape) != list(add_sp.spatial_shape)) or (base_sp.batch_size != add_sp.batch_size):
            return base_sp

        # indices: [N, 4] = [batch, z, y, x]
        Z, Y, X = base_sp.spatial_shape
        stride_y = X
        stride_z = Y * X
        stride_b = Z * Y * X

        base_idx = base_sp.indices.to(dtype=torch.long)
        add_idx = add_sp.indices.to(dtype=torch.long)
        base_lin = base_idx[:, 0] * stride_b + base_idx[:, 1] * stride_z + base_idx[:, 2] * stride_y + base_idx[:, 3]
        add_lin = add_idx[:, 0] * stride_b + add_idx[:, 1] * stride_z + add_idx[:, 2] * stride_y + add_idx[:, 3]

        # Sort add coordinates, then search them from the base tensor.
        order = torch.argsort(add_lin)
        add_lin_sorted = add_lin[order]
        add_feat_sorted = add_sp.features[order]

        pos = torch.searchsorted(add_lin_sorted, base_lin)
        # pos can equal numel; guard that case before indexing.
        valid = pos < add_lin_sorted.numel()
        hit = valid.clone()
        if valid.any():
            hit[valid] = add_lin_sorted[pos[valid]] == base_lin[valid]

        # Add features only at matched coordinates.
        add_to_base = base_sp.features.new_zeros(base_sp.features.shape)
        if hit.any():
            add_to_base[hit] = add_feat_sorted[pos[hit]]

        return replace_feature(base_sp, base_sp.features + add_to_base)

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        pc_range = model_cfg.PC_RANGE
        voxel_size = model_cfg.VOXEL_SIZE
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1_input_channel = 16
        self.conv1_output_channel = 16
        self.conv2_input_channel = 16
        self.conv2_output_channel = 32
        self.conv3_input_channel = 32
        self.conv3_output_channel = 64
        self.conv4_input_channel = 64
        self.conv4_output_channel = 64

        img_channel = 256
        img_levels = [0, 1, 2, 3]

        self.img_levels = img_levels
        img_channels_fpn = [img_channel] * len(img_levels)
        mid_channels = [32, 64, 128, 128]
        norm_2d_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.lateral_convs_fpn = nn.ModuleList()
        for i in range(len(img_channels_fpn)):
            l_conv = nn.Sequential(
                nn.Conv2d(img_channels_fpn[i], mid_channels[i], 3, padding=1),
                norm_2d_layer(mid_channels[i]),
                nn.ReLU(inplace=False)
            )
            self.lateral_convs_fpn.append(l_conv)

        # ================= ENCODER =================
        self.conv1 = spconv.SparseSequential(
            block(self.conv1_input_channel, self.conv1_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm1'),
        )
        self.fuse_layer1 = PAAF(self.conv1_output_channel, mid_channels[0], self.conv2_input_channel, 1, 1, pc_range=pc_range, voxel_size=voxel_size)

        self.conv2 = spconv.SparseSequential(
            block(self.conv2_input_channel, self.conv2_output_channel, 3, norm_fn=norm_fn, stride=2, padding=1,
                  indice_key='spconv2', conv_type='spconv'),
            block(self.conv2_output_channel, self.conv2_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm2'),
            block(self.conv2_output_channel, self.conv2_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm2'),
        )
        self.fuse_layer2 = PAAF(self.conv2_output_channel, mid_channels[1], self.conv3_input_channel, 2, 2, pc_range=pc_range, voxel_size=voxel_size)

        self.conv3 = spconv.SparseSequential(
            block(self.conv3_input_channel, self.conv3_output_channel, 3, norm_fn=norm_fn, stride=2, padding=1,
                  indice_key='spconv3', conv_type='spconv'),
            block(self.conv3_output_channel, self.conv3_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm3'),
            block(self.conv3_output_channel, self.conv3_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm3'),
        )
        self.fuse_layer3 = PAAF(self.conv3_output_channel, mid_channels[2], self.conv4_input_channel, 4, 3, pc_range=pc_range, voxel_size=voxel_size)

        self.conv4 = spconv.SparseSequential(
            block(self.conv4_input_channel, self.conv4_output_channel, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1),
                  indice_key='spconv4', conv_type='spconv'),
            block(self.conv4_output_channel, self.conv4_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm4'),
            block(self.conv4_output_channel, self.conv4_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm4'),
        )
        self.fuse_layer4 = PAAF(self.conv4_output_channel, mid_channels[3], self.conv4_input_channel, 8, 4, pc_range=pc_range, voxel_size=voxel_size)

        # ================= DECODER =================
        self.deconv1 = spconv.SparseSequential(
            block(self.conv4_output_channel, self.conv4_input_channel, 3, norm_fn=norm_fn, indice_key='spconv4',
                  conv_type='inverseconv'),
            block(self.conv4_input_channel, self.conv4_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm1'),
            block(self.conv4_input_channel, self.conv4_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm1'),
        )
        self.de_fuse_layer1 = PAAF(self.conv4_input_channel, mid_channels[2], self.conv3_output_channel, 4, 3, pc_range=pc_range, voxel_size=voxel_size)

        # Downsample decoder stride-4 features to stride 8 and add them onto x_fuse4.
        # spconv stride conv cannot reuse an existing indice_key, so this block uses a new key.
        self.down_fuse_from_s4_to_s8 = spconv.SparseSequential(
            block(self.conv3_output_channel, self.conv4_output_channel, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1),
                  indice_key='spconv4_fuse', conv_type='spconv'),
        )

        self.deconv2 = spconv.SparseSequential(
            block(self.conv3_output_channel, self.conv3_input_channel, 3, norm_fn=norm_fn, indice_key='spconv3',
                  conv_type='inverseconv'),
            block(self.conv3_input_channel, self.conv3_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm2'),
            block(self.conv3_input_channel, self.conv3_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm2'),
        )
        self.de_fuse_layer2 = PAAF(self.conv3_input_channel, mid_channels[1], self.conv2_output_channel, 2, 2, pc_range=pc_range, voxel_size=voxel_size)

        self.deconv3 = spconv.SparseSequential(
            block(self.conv2_output_channel, self.conv2_input_channel, 3, norm_fn=norm_fn, indice_key='spconv2',
                  conv_type='inverseconv'),
            block(self.conv2_input_channel, self.conv2_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm3'),
            block(self.conv2_input_channel, self.conv2_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm3'),
        )
        # x_defuse3 uses conv1_output_channel (16).
        self.de_fuse_layer3 = PAAF(self.conv2_input_channel, mid_channels[0], self.conv1_output_channel, 1, 1, pc_range=pc_range, voxel_size=voxel_size)

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)

        # Keep the stride-8 tensor expected by the Voxel R-CNN head.
        # conv_out only downsamples Z so HeightCompression yields 512 BEV channels.
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        
        # Channel map consumed by the second-stage RoI head.
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        img_feats_fpn = batch_dict['image_fpn']

        # Project each FPN level to the channel width expected by PAAF.
        # Resolution is handled inside PAAF, so no extra interpolate is applied here.
        batch_dict["img_fpn_feats_ori_shape"] = [
            lateral_conv(img_feats_fpn[i])
            for i, lateral_conv in zip(self.img_levels, self.lateral_convs_fpn)
        ]

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        # ====== Encoder Path ======
        x_conv1 = self.conv1(x)
        x_fuse1 = self.fuse_layer1(x_conv1, batch_dict, 1)

        x_conv2 = self.conv2(x_fuse1)
        x_fuse2 = self.fuse_layer2(x_conv2, batch_dict, 1)

        x_conv3 = self.conv3(x_fuse2)
        x_fuse3 = self.fuse_layer3(x_conv3, batch_dict, 1)

        x_conv4 = self.conv4(x_fuse3)
        x_fuse4 = self.fuse_layer4(x_conv4, batch_dict, 1)

        # ====== Decoder Path ======
        x_deconv1 = self.deconv1(x_fuse4)
        x_defuse1 = self.de_fuse_layer1(x_deconv1, batch_dict, 1)
        x_defuse1 = x_defuse1 + x_fuse3  # Skip Connection

        x_deconv2 = self.deconv2(x_defuse1)
        x_defuse2 = self.de_fuse_layer2(x_deconv2, batch_dict, 1)
        x_defuse2 = x_defuse2 + x_fuse2  # Skip Connection

        x_deconv3 = self.deconv3(x_defuse2)
        x_defuse3 = self.de_fuse_layer3(x_deconv3, batch_dict, 1)
        x_defuse3 = x_defuse3 + x_fuse1  # Skip Connection (Stride 1 Feature)

        # Inject stride-4 decoder detail into the stride-8 encoder feature.
        x_s4_to_s8 = self.down_fuse_from_s4_to_s8(x_defuse1)
        x_fuse4 = self._sparse_add_by_coords(x_fuse4, x_s4_to_s8)

        # for detection head: stride=8 branch
        out = self.conv_out(x_fuse4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })

        # Multi-scale sparse tensors for the refinement head.
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_defuse3,
                'x_conv2': x_defuse2,
                'x_conv3': x_defuse1,
                'x_conv4': x_fuse4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict