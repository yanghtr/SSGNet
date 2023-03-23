from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch_scatter

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import box_utils
from ...utils import loss_utils
from ...utils import common_utils

from ..backbones_2d.unet_2d import UNet2D, DoubleConv, UNet_Dense2D_Lite
from .implicit_backbone import ImplicitNet2d
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


class FCBlock(nn.Module):
    def __init__(self, fc_cfg, input_channels, output_channels=None, with_last_layer=True):
        """
        Args:
            fc_cfg: list of hidden features
            with_last_layer:
                if with_last_layer == True:
                    last layer is simply nn.Linear, need to specify output_channels
                else:
                    last layer is appended with BN and ReLU, use fc_cfg[-1] as output_channels
        Returns:
            fc_module: input: (*, input_channels), output: (*, output_channels)
        """
        super().__init__()
        if with_last_layer:
            assert(output_channels is not None)
        else:
            assert(output_channels is None)
            output_channels = fc_cfg[-1]

        fc_layers = []
        c_in = input_channels
        for k in range(0, len(fc_cfg) - 1):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]

        if with_last_layer:
            fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        else:
            fc_layers.extend([
                nn.Linear(c_in, output_channels, bias=False),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(),
            ])

        self.model = nn.Sequential(*fc_layers)
        self.out_dim = output_channels

    def forward(self, x):
        return self.model(x)


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, inplanes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(inplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
    
    
class PillarRes18BackBone8xHybridRefineBEV(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = torch.FloatTensor(point_cloud_range).cuda()
        self.grid_size = torch.FloatTensor(grid_size).cuda()
        self.voxel_size = torch.FloatTensor(voxel_size).cuda()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block
        dense_block = post_act_block_dense
        
        norm_fn_dense = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        # XXXXXX Explicit BEV Segmentation START
        self.conv_collapse_exp = block(128, 128, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv_collapse_exp', conv_type='spconv')
        self.unet2d = UNet2D(128, 1)
        self.conv_mask_exp = DoubleConv(1, 128)
        self.conv_merge_mask_exp = nn.Sequential(
            dense_block(256, 128, 3, stride=1, padding=1, norm_fn=norm_fn_dense),
            dense_block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn_dense),
        )
        # XXXXXX Explicit BEV Segmentation END

        # XXXXXX Implicit BEV Segmentation START
        self.conv_collapse_imp = block(128, 128, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv_collapse_imp', conv_type='spconv')
        self.conv_down_3to4_imp = nn.Sequential(
            dense_block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn_dense),
            dense_block(128, 128, 3, stride=1, padding=1, norm_fn=norm_fn_dense),
        )
        self.implicit_net2d = ImplicitNet2d({}, 128, 1, grid_size[:2], voxel_size[:2], point_cloud_range, stride_2d=1)
        self.conv_mask_imp = DoubleConv(1, 128)
        self.conv_merge_mask_imp = nn.Sequential(
            dense_block(256, 128, 3, stride=1, padding=1, norm_fn=norm_fn_dense),
        )
        # XXXXXX Implicit BEV Segmentation END

        self.conv_merge_bev = nn.Sequential(
            dense_block(128*3, 128, 3, stride=1, padding=1, norm_fn=norm_fn_dense),
        )

        self.num_point_features = 128
        self.num_bev_features = 128
        self.forward_ret_dict = {}

        self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)

    @staticmethod
    def sigmoid(x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['voxel_coords'][:, [0, 2, 3]]
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Waymo: x_conv1: [1504, 1504], x_conv2: [752, 752], x_conv3: [376, 376], x_conv4: dense(188, 188), x_conv5: dense(94, 94)
        # # IMPORTANT NOTE: initial update in order to use self.sample_grid_centers, will be updated later
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv3': input_sp_tensor,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv3': 1,
            }
        })

        # Explicit BEV Segmentation
        x_conv3_exp = self.conv_collapse_exp(input_sp_tensor)
        x_conv3_dense_exp = x_conv3_exp.dense() # (B, 128, 464, 464)
        bev_seg_logits3_exp = self.unet2d(x_conv3_dense_exp) # (B, 1, 464, 464)
        bev_seg_prob3_exp = self.sigmoid(bev_seg_logits3_exp) # NOTE: Have to use self.sigmoid, otherwise NaN or Inf found in input tensor
        self.forward_ret_dict['bev_seg_prob3_exp'] = bev_seg_prob3_exp
        batch_dict['bev_seg_prob3_exp'] = bev_seg_prob3_exp

        bev_seg_mask_feats3_exp = self.conv_mask_exp(bev_seg_prob3_exp) # (B, 128, 464, 464)
        bev_feats3_exp = torch.cat((x_conv3_dense_exp, bev_seg_mask_feats3_exp), dim=1)
        bev_feats3to4_exp = self.conv_merge_mask_exp(bev_feats3_exp)

        # Implicit BEV Segmentation
        x_conv3_imp = self.conv_collapse_imp(input_sp_tensor) # (B, 128, 464, 464)
        x_conv3_dense_imp = x_conv3_imp.dense() # (B, 128, 464, 464)
        x_conv3to4_imp = self.conv_down_3to4_imp(x_conv3_dense_imp) # (B, 128, 464, 464), map to latent space

        sample_grid_points = self.sample_grid_centers(batch_dict, bev_seg_layer='x_conv3') # (H, W, 3)
        H, W = sample_grid_points.shape[:2]
        sample_grid_points_2d = sample_grid_points[:, :, :2].reshape(1, -1, 2).expand(batch_size, -1, -1) # (B, H*W, 2)

        bev_seg_logits4_imp = self.implicit_net2d(x_conv3to4_imp, sample_grid_points_2d) # (B, H*W, 1)
        bev_seg_prob4_imp = self.sigmoid(bev_seg_logits4_imp) # NOTE: Have to use self.sigmoid, otherwise NaN or Inf found in input tensor
        bev_seg_prob4_imp = bev_seg_prob4_imp.reshape(batch_size, 1, H, W) # (B, 1, H, W)

        batch_dict['bev_seg_prob4_imp'] = bev_seg_prob4_imp # (B, 1, H, W)

        if self.training:
            sample_points_3d, obj_points_list, scene_points_list = self.sample_points_surface(batch_dict, pn_ratio=0.75, p_ratio=2/3, Ns=65536) # (B, Ns, 3)
            sample_points_2d = sample_points_3d[:, :, :2] # (B, Ns, 2)

            smp_seg_logits4_imp = self.implicit_net2d(x_conv3to4_imp, sample_points_2d) # (B, Ns, 1)
            smp_seg_prob4_imp = self.sigmoid(smp_seg_logits4_imp) # NOTE: Have to use self.sigmoid, otherwise NaN or Inf found in input tensor

            self.forward_ret_dict['smp_seg_logits4_imp'] = smp_seg_logits4_imp # (B, Ns, 1)
            batch_dict['smp_seg_prob4_imp'] = smp_seg_prob4_imp
            batch_dict['sample_points_3d'] = sample_points_3d

        bev_seg_mask_feats4_imp = self.conv_mask_imp(bev_seg_prob4_imp) # (B, 128, 464, 464)
        bev_feats4_imp = torch.cat((x_conv3to4_imp, bev_seg_mask_feats4_imp), dim=1) # (B, 256, 464, 464)
        bev_feats4_imp = self.conv_merge_mask_imp(bev_feats4_imp) # (B, 256, 464, 464)

        # Merge dense features from all branches
        x_conv4 = input_sp_tensor.dense()
        x_conv4 = torch.cat((x_conv4, bev_feats3to4_exp, bev_feats4_imp), dim=1) # (B, 128*3, 464, 464)
        x_conv4 = self.conv_merge_bev(x_conv4) # (B, 128, 464, 464)

        batch_dict.update({
            'spatial_features': x_conv4,
            'spatial_features_stride': 1
        })

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv3': input_sp_tensor,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv3': 1,
                'x_conv4': 1,
            }
        })

        if self.training:
            targets_dict_imp = self.assign_targets_imp(batch_dict)
            targets_dict_exp = self.assign_targets_exp(batch_dict, bev_seg_layer='x_conv3')
            self.forward_ret_dict.update(targets_dict_imp)
            self.forward_ret_dict.update(targets_dict_exp)

        return batch_dict


    def sample_grid_centers(self, batch_dict, bev_seg_layer):
        """ Get grid centers
        Args:
        Returns:
            bev_voxel_centers: (H, W, 3), z coordinates use middle of z of point_cloud_range
        """
        # NOTE: the following two lines are different between 2D and 3D
        stride_2d = batch_dict['multi_scale_2d_strides'][bev_seg_layer]
        spatial_shape = batch_dict['multi_scale_2d_features'][bev_seg_layer].spatial_shape # HW -> HW

        voxel_size_s = self.voxel_size[:2] * stride_2d
        grid_size_s = (self.grid_size[:2] / stride_2d).int() # XY
        scale_xy = grid_size_s[0] * grid_size_s[1]
        scale_y = grid_size_s[1]
        assert(grid_size_s.cpu().numpy().tolist() == spatial_shape[::-1])

        bev_x = torch.arange(grid_size_s[0]).cuda()
        bev_y = torch.arange(grid_size_s[1]).cuda()
        grid_x, grid_y = torch.meshgrid(bev_x, bev_y, indexing='xy')
        bev_pixel_coords = torch.stack((grid_x, grid_y), dim=-1) # (H, W, 2)
        range_z_center = (self.point_cloud_range[2] + self.point_cloud_range[5]) / 2
        bev_pixel_centers = (bev_pixel_coords + 0.5) * voxel_size_s + self.point_cloud_range[:2]
        bev_voxel_centers = torch.cat((bev_pixel_centers, torch.ones_like(bev_pixel_centers[:, :, 0:1]) * range_z_center), dim=-1) # (H, W, 3)

        return bev_voxel_centers


    def sample_points_surface(self, batch_dict, pn_ratio=0.5, p_ratio=2/3, Ns=80000):
        """ Sample points in raw scene scale
        Args:
            pn_ratio: pn_ratio points around the gt boxes, (1 - pn_ratio) points randomly in the scene
            p_ratio: in pn_ratio points, p_ratio points are positives, (1 - pn_ratio) are negatives
            Ns: total num of sample points in the scenes
        Returns:
            all_points: (B, Ns, 3)
        """
        gt_boxes = batch_dict['gt_boxes'] # (B, N, 8)
        batch_size = batch_dict['batch_size']

        assert(p_ratio < 1)
        assert(gt_boxes.shape[-1] == 8)

        large_gt_boxes = gt_boxes.clone() # (B, N, 8)
        extra_width_s = np.sqrt(1 / p_ratio) - 1
        extra_sizes = gt_boxes[..., 3:6] * extra_width_s
        large_gt_boxes[..., 3:6] += extra_sizes

        all_points_list = []
        obj_points_list = []
        scene_points_list = []
        for b in range(batch_size):
            boxes3d = large_gt_boxes[b] # (N, 8)
            boxes3d_mask = boxes3d[:, -1] > 0
            num_box = torch.sum(boxes3d_mask).item()
            boxes3d = boxes3d[:num_box] # (num_box, 8)

            num_point_per_box = int(pn_ratio * Ns / num_box)
            num_scene_sample = Ns - num_point_per_box * num_box

            obj_points = torch.rand((num_box, num_point_per_box, 3)) - 0.5
            obj_points = obj_points.to(gt_boxes.device) # (num_box, num_point_per_box, 3), [-0.5, 0.5]
            obj_points = boxes3d[:, None, 3:6].repeat(1, num_point_per_box, 1) * obj_points # (num_box, num_point_per_box, 3)
            obj_points = common_utils.rotate_points_along_z(obj_points, boxes3d[:, 6])
            obj_points += boxes3d[:, None, 0:3] # (num_box, num_point_per_box, 3)

            scene_points = torch.rand((num_scene_sample, 3)).to(gt_boxes.device) # (num_scene_sample, 3), [0, 1]
            scene_points = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) * scene_points + self.point_cloud_range[0:3] # (num_scene_sample, 3)

            all_points_b = torch.cat((obj_points.reshape(-1, 3), scene_points), dim=0) # (Ns, 3)
            all_points_list.append(all_points_b)
            obj_points_list.append(obj_points)
            scene_points_list.append(scene_points)

        all_points = torch.stack(all_points_list) # (B, Ns, 3)
        return all_points, obj_points_list, scene_points_list


    def assign_targets_imp(self, batch_dict):

        sample_points_3d = batch_dict['sample_points_3d'] # (B, Ns, 3)
        batch_size = batch_dict['batch_size']
        gt_boxes = batch_dict['gt_boxes'] # (B, Nbox, 8)

        range_z_center = (self.point_cloud_range[2] + self.point_cloud_range[5]) / 2
        sample_points_3d[:, :, 2] = range_z_center # (B, Ns, 3)

        # increase z size to ensure all 2D points are included
        extra_width = [0, 0, 0]
        extra_width[2] = (self.point_cloud_range[5] - self.point_cloud_range[2]).item()
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1]) # (B, max_num_boxes, 8)

        box_ids_of_pts = points_in_boxes_gpu(sample_points_3d, extend_gt_boxes[:, :, :7]) # (B, Ns)
        sample_sem_gt = (box_ids_of_pts >= 0).float() # (B, Ns)

        targets_dict = {
            'sample_sem_gt': sample_sem_gt,
        }
        return targets_dict


    def assign_targets_exp(self, batch_dict, bev_seg_layer):

        # NOTE: the following two lines are different between 2D and 3D
        stride_2d = batch_dict['multi_scale_2d_strides'][bev_seg_layer]
        spatial_shape = batch_dict['multi_scale_2d_features'][bev_seg_layer].spatial_shape # HW -> HW

        voxel_size_s = self.voxel_size[:2] * stride_2d
        grid_size_s = (self.grid_size[:2] / stride_2d).int() # XY
        scale_xy = grid_size_s[0] * grid_size_s[1]
        scale_y = grid_size_s[1]
        assert(grid_size_s.cpu().numpy().tolist() == spatial_shape[::-1])

        batch_size = batch_dict['batch_size']
        gt_boxes = batch_dict['gt_boxes']

        extra_width = self.voxel_size.detach().cpu().numpy() * stride_2d / 2
        extra_width[2] = self.point_cloud_range[5] - self.point_cloud_range[2]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=extra_width
        ).view(batch_size, -1, gt_boxes.shape[-1]) # (B, max_num_boxes, 8)

        bev_x = torch.arange(grid_size_s[0]).cuda()
        bev_y = torch.arange(grid_size_s[1]).cuda()
        grid_x, grid_y = torch.meshgrid(bev_x, bev_y, indexing='xy')
        bev_pixel_coords = torch.stack((grid_x, grid_y), dim=-1) # (H, W, 2)
        range_z_center = (self.point_cloud_range[2] + self.point_cloud_range[5]) / 2
        bev_pixel_centers = (bev_pixel_coords + 0.5) * voxel_size_s + self.point_cloud_range[:2]
        bev_voxel_centers = torch.cat((bev_pixel_centers, torch.ones_like(bev_pixel_centers[:, :, 0:1]) * range_z_center), dim=-1) # (H, W, 3)

        bev_points = bev_voxel_centers.reshape(1, -1, 3).repeat(batch_size, 1, 1) # (1, H*W, 3) -> (B, H*W, 3)
        box_ids_of_pts = points_in_boxes_gpu(bev_points, extend_gt_boxes[:, :, :7]) # (B, H*W)
        bev_sem = (box_ids_of_pts >= 0).float() # (B, H*W)

        bev_seg_gt = bev_sem.reshape(batch_size, spatial_shape[0], spatial_shape[1])
        gaussian_blurrer = T.GaussianBlur(kernel_size=(3, 3))
        bev_seg_gt_blur = gaussian_blurrer(bev_seg_gt)
        bev_seg_gt = torch.maximum(bev_seg_gt, bev_seg_gt_blur)

        targets_dict = {
            'bev_seg_gt': bev_seg_gt,
        }
        return targets_dict


    @staticmethod
    def get_cls_layer_loss(point_cls_preds, point_cls_labels, cls_loss_func):
        """
        Args:
            point_cls_preds: (N, num_class)
            point_cls_labels: (N,), positives: > 0 (each class: 1, 2, 3...), negative: == 0
        """
        assert(point_cls_preds.shape[0] == point_cls_labels.shape[0])
        assert(len(point_cls_preds.shape) == 2 and len(point_cls_labels.shape) == 1)
        num_class = point_cls_preds.shape[1]

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0) # 1 / P, sum = (N + P) / P

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:] # (N, num_class), 0 is negative, [..., 1:] selects all the positives
        cls_loss_src = cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        return point_loss_cls

    def get_loss(self, tb_dict=None):

        smp_seg_logits4_imp = self.forward_ret_dict['smp_seg_logits4_imp'] # (B, Ns, 1)
        sample_sem_gt = self.forward_ret_dict['sample_sem_gt'] # (B, Ns)
        sample_loss = self.get_cls_layer_loss(smp_seg_logits4_imp.reshape(-1, 1), sample_sem_gt.reshape(-1), self.cls_loss_func)
        sample_loss *= self.model_cfg.IMP_SEG.LOSS_CONFIG.LOSS_WEIGHTS['smp_weight']

        bev_seg_prob3_exp = self.forward_ret_dict['bev_seg_prob3_exp'] # (B, 1, 376, 376)
        bev_seg_gt = self.forward_ret_dict['bev_seg_gt'][:, None, :, :] # (B, 1, H, W)
        bev_loss = loss_utils.neg_loss_cornernet(bev_seg_prob3_exp, bev_seg_gt)
        bev_loss *= self.model_cfg.EXP_SEG.LOSS_CONFIG.LOSS_WEIGHTS['bev_weight']

        loss = sample_loss + bev_loss

        if tb_dict is None:
            tb_dict = {}
        tb_dict['sample_loss'] = sample_loss.item()
        tb_dict['bev_loss'] = bev_loss.item()
        return loss, tb_dict


