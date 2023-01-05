from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
from torchpack import distributed as dist


from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
    build_loss,
    build_projector
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
import os

from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        calico: Dict[str, Any],
        save_dir: str,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        if calico is not None:
            ##TODO: add calico
            self.pretraining = True
            if calico["lidar_projector"]["type"] == "SharedProjector":
                self.lidar_projector = build_projector(calico["lidar_projector"])
                self.camera_projector = self.lidar_projector
            else:
                self.lidar_projector = build_projector(calico["lidar_projector"])
                self.camera_projector = build_projector(calico["camera_projector"])
            from torchvision.ops import RoIAlign
            self.roi_align = RoIAlign(**calico["roi_align"])
            calico["loss"]["gather_with_grad"] = True
            calico["loss"]["cache_labels"] = False
            calico["loss"]["rank"] = dist.rank()
            calico["loss"]["world_size"] = dist.size()
            self.pretrain_loss = build_loss(calico["loss"])
        else:
            self.pretraining = False
            self.decoder = nn.ModuleDict(
                {
                    "backbone": build_backbone(decoder["backbone"]),
                    "neck": build_neck(decoder["neck"]),
                }
            )
            self.heads = nn.ModuleDict()
            for name in heads:
                if heads[name] is not None:
                    self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            if self.pretraining:
                self.loss_scale = dict()
                self.loss_scale["calico"] = 1.0
            else:
                self.loss_scale = dict()
                for name in heads:
                    if heads[name] is not None:
                        self.loss_scale[name] = 1.0


        self.init_weights()
        self.counter = 0

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()
        

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            # print(k,res)
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points","points_2"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        points_2=None,
        lidar_aug_matrix_2=None,
        pooled_bbox=None,
        pooled_bbox_2=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):  
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                pooled_bbox,
                pooled_bbox_2,
                lidar_aug_matrix_2,
                points_2,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points","points_2"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        pooled_bbox=None,
        pooled_bbox_2=None,
        lidar_aug_matrix_2=None,
        points_2=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):  
        # ##### TEST CORRECTNESS ### TODO: remove me later
        # from calico_tools.visualize.general import draw_pointcloud_polygon_matplotlib
        # pooled_bbox[0][:,0::2] = pooled_bbox[0][:,0::2] * 0.075 - 54.0
        # pooled_bbox[0][:,1::2] = pooled_bbox[0][:,1::2] * 0.075 - 54.0
        # draw_pointcloud_polygon_matplotlib(points[0].cpu().numpy(), bboxes = pooled_bbox[0].cpu().numpy(),save='./data/temp_test/'+str(self.counter)+'.png')
        # self.counter += 1
        # #####################
        features = []
        feature_2 = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                print("I am here")
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
                if points_2 is not None:
                    print("I am here")
                    feature_camera_2 = self.extract_camera_features(
                        img,
                        points_2,
                        camera2ego,
                        lidar2ego,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix_2,
                        metas,
                    )
                    feature_2.append(feature_camera_2)

            elif sensor == "lidar":
                print("points",points[0].shape)
                feature = self.extract_lidar_features(points)
                if points_2 is not None:
                    feature_lidar_2 = self.extract_lidar_features(points_2)
                    feature_2.append(feature_lidar_2)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training and not self.pretraining:
            # avoid OOM
            features = features[::-1]
        batch_size = features[0].shape[0]

        if not self.pretraining:
            if self.fuser is not None:
                x = self.fuser(features)
            else:
                assert len(features) == 1, features
                x = features[0]
            x = self.decoder["backbone"](x)
            x = self.decoder["neck"](x)

        if self.training:
            self.counter = 0
            outputs = {}
            if self.pretraining:
                # number_bbox = pooled_bbox[0].shape[0]
                roi_lidar_feature = self.roi_align(features[0], pooled_bbox)
                roi_camera_feature = self.roi_align(features[1], pooled_bbox)
                projected_lidar_feature = self.lidar_projector(roi_lidar_feature,'lidar')
                projected_camera_feature = self.camera_projector(roi_camera_feature,'camera')
                ##L2 normalize################
                normalized_projected_lidar_feaure = F.normalize(projected_lidar_feature, p=2, dim=1)
                normalized_projected_camera_feature = F.normalize(projected_camera_feature, p=2, dim=1)
                ##############################
                loss1 = self.pretrain_loss(normalized_projected_camera_feature,normalized_projected_lidar_feaure, 10.0)
                outputs['loss/pretrain/calico_view_1_lc'] = loss1

                roi_lidar_feature_2 = self.roi_align(feature_2[0], pooled_bbox_2)
                roi_camera_feature_2 = self.roi_align(feature_2[1], pooled_bbox_2)
                projected_lidar_feature_2 = self.lidar_projector(roi_lidar_feature_2,'lidar')
                projected_camera_feature_2 = self.camera_projector(roi_camera_feature_2,'camera')
                ##L2 normalize################
                normalized_projected_lidar_feaure_2 = F.normalize(projected_lidar_feature_2, p=2, dim=1)
                normalized_projected_camera_feature_2 = F.normalize(projected_camera_feature_2, p=2, dim=1)
                ##############################
                loss2 = self.pretrain_loss(normalized_projected_camera_feature_2,normalized_projected_lidar_feaure_2, 10.0)
                outputs['loss/pretrain/calico_view_2_lc'] = loss2

                loss3 = self.pretrain_loss(normalized_projected_camera_feature,normalized_projected_camera_feature_2, 10.0)
                outputs['loss/pretrain/calico_view_12_cc'] = loss3

                loss4 = self.pretrain_loss(normalized_projected_lidar_feaure,normalized_projected_lidar_feaure_2, 10.0)
                outputs['loss/pretrain/calico_view_12_ll'] = loss4

            else:
                for type, head in self.heads.items():
                    if type == "object":
                        pred_dict = head(x, metas)
                        losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    elif type == "map":
                        losses = head(x, gt_masks_bev)
                    else:
                        raise ValueError(f"unsupported head: {type}")
                    for name, val in losses.items():
                        if val.requires_grad:
                            outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                        else:
                            outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            if self.pretraining:
                if self.counter % 50 == 0:
                    import torchvision
                    gray_scale_1 = torch.sum(features[0].squeeze(),0)
                    gray_scale_1 = gray_scale_1 / features[0].shape[0]
                    gray_scale_1 = ((gray_scale_1 - gray_scale_1.min()) / (gray_scale_1.max() - gray_scale_1.min()) * 255).to("cpu",torch.uint8) 

                    gray_scale_2 = torch.sum(features[1].squeeze(),0)
                    gray_scale_2 = gray_scale_2 / features[1].shape[0]
                    gray_scale_2 = ((gray_scale_2 - gray_scale_2.min()) / (gray_scale_2.max() - gray_scale_2.min()) * 255).to("cpu",torch.uint8)

                    img1=torchvision.utils.draw_bounding_boxes(gray_scale_1.unsqueeze(0),pooled_bbox[0]//8,colors="red") / 255.#.numpy()
                    img2=torchvision.utils.draw_bounding_boxes(gray_scale_2.unsqueeze(0),pooled_bbox[0]//8,colors="red") / 255.#.numpy()
                    # saved_image = torchvision.utils.make_grid([img1,img2], nrow=1)
                    torchvision.utils.save_image([img1,img2], os.path.join(self.save_dir,str(self.counter)+'.png'))
                self.counter += 1
                return outputs
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
