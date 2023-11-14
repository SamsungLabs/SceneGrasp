import sys
import os
from pathlib import Path
import time
import argparse
import torch
import numpy as np
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance
from common.config import config_global_paths
from common.config.config_dataset_details import (
    get_emb_dim,
    get_grasp_loss_topk,
    get_min_chamfer_loss_for_grasp_losses,
    get_n_points,
    get_num_cpus,
    get_shape_grasp_ae_initial_lr,
    get_shape_grasp_ae_logs_prefix,
    get_weight_shape_loss,
)
from scene_grasp.scale_shape_grasp_ae.model.auto_encoder_scale_grasp import (
    PointCloudScaleBasedGraspAE,
)
from scene_grasp.scale_shape_grasp_ae.dataset.shape_dataset_grasp_map import (
    ShapeDatasetGraspMap,
)
from common.utils.shape_utils import init_logs_dir


def get_control_point_tensor(symmetric=False):
    control_points = np.load("centersnap/external/shape_pretraining/panda.npy")[:, :3]
    if symmetric:
        control_points = [
            [0, 0, 0],
            control_points[1, :],
            control_points[0, :],
            control_points[-1, :],
            control_points[-2, :],
        ]
    else:
        control_points = [
            [0, 0, 0],
            control_points[0, :],
            control_points[1, :],
            control_points[-2, :],
            control_points[-1, :],
        ]
    control_points = np.asarray(control_points, dtype=np.float32)
    return control_points


def train_net(args):
    # Setup environment
    logs_root = config_global_paths.PROJECT_LOGS_ROOT / get_shape_grasp_ae_logs_prefix()
    logs_dir, logger = init_logs_dir(
        logs_root,
        logger_name="train_log",
        logs_dir_suffix=args.logs_dir_suffix,
        input_args=vars(args),
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # model & loss
    estimator = PointCloudScaleBasedGraspAE(
        args.emb_dim, args.num_point, args.choose_bd_sign
    )
    estimator.cuda()
    if args.resume_model is not None:
        logger.info(f"Resuming model from checkpoint: {args.resume_model}")
        estimator.load_state_dict(torch.load(args.resume_model))

    # dataset
    train_dataset = ShapeDatasetGraspMap(
        args.dataset_root,
        mode="train",
        augment=False,
        n_points=args.num_point,
        category_id=args.category_id,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataset = ShapeDatasetGraspMap(
        args.dataset_root,
        mode="val",
        augment=False,
        n_points=args.num_point,
        category_id=args.category_id,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # train
    st_time = time.time()
    # global_step is roughly how many total batches we have optimized on till now
    global_step = (
        ((train_dataset.length + args.batch_size - 1) // args.batch_size)
        * args.repeat_epoch
        * (args.start_epoch - 1)
    )
    logger.info(f"train_dataset {train_dataset.length}")
    logger.info(f"val dataset {val_dataset.length}")
    logger.info(f"global step {global_step}")
    decay_count = -1

    MIN_CHAMFER_LOSS_FOR_GRASP_EST = args.min_chamfer_loss_for_grasp_losses
    PROPAGATION_RADIUS = (
        1e-2  # How close two points should be to propogate grasp-success
    )
    GRASP_WIDTH_BIN_WEIGHTS = (
        torch.tensor(
            [
                0.16652107,
                0.21488856,
                0.37031708,
                0.55618503,
                0.75124664,
                0.93943357,
                1.07824539,
                1.19423112,
                1.55731375,
                3.17161779,
            ]
        )
        .cuda()
        .to(torch.float)
    )
    MAX_GRIPPER_WIDTH = 0.08
    GRIPPER_DEPTH = 0.1034
    assert 0 <= args.weight_shape_loss and args.weight_shape_loss <= 1
    WEIGHT_SHAPE_LOSS = args.weight_shape_loss
    WEIGHT_GRASP_PARAMETERS = (1 - args.weight_shape_loss) * 1e-2
    WEIGHT_GRASP_SUCCESS_LOSS = 1
    WEIGHT_GRASP_WIDTH_LOSS = 1
    WEIGHT_GRASP_ORI_LOSS = 10

    GRASP_LOSS_TOP_K = args.grasp_loss_topk
    gripper_control_points = get_control_point_tensor()
    gripper_control_points = torch.tensor(gripper_control_points).cuda().to(torch.float)
    gripper_control_points_sym = get_control_point_tensor(symmetric=True)
    gripper_control_points_sym = (
        torch.tensor(gripper_control_points_sym).cuda().to(torch.float)
    )
    # Ok. here let me quickly define the gripper points.
    current_lr = None
    for epoch in range(args.start_epoch, args.max_epoch + 1):
        # train one epoch
        log_time = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))
        logger.info(f"Time {log_time}, Epoch {epoch}, Training started")
        # create optimizer and adjust learning rate if needed
        if global_step // args.decay_step > decay_count:
            decay_count += 1
            if decay_count < len(args.decay_rate):
                current_lr = args.lr * args.decay_rate[decay_count]
                optimizer = torch.optim.Adam(estimator.parameters(), lr=current_lr)
                logger.info(f"Changed learning rate to {current_lr}")
        epoch_batch_idx = 0
        estimator.train()
        logger.info(f"- Learning rate: {current_lr}")
        for _ in range(args.repeat_epoch):
            for _, data in enumerate(train_dataloader):
                inp, batch_target = data
                batch_xyz = inp["xyz"][:, :, :3].cuda().to(torch.float)  # (B, N, 3)
                batch_scales = inp["scale"].cuda().to(torch.float)
                batch_target = {
                    key: value.cuda().to(torch.float)
                    for key, value in batch_target.items()
                }
                gt_batch_gw = batch_target["grasp_width_one_hot"]  # (B, N, 10)
                gt_batch_gwv = batch_target["grasp_width"]  # (B, N)
                gt_batch_a = batch_target["approach_dir"]  # (B, N, 3)
                gt_batch_b = batch_target["baseline_dir"]  # (B, N, 3)

                optimizer.zero_grad()
                _, endpoints = estimator(batch_xyz, batch_scales)
                pred_batch_xyz = endpoints["xyz"]  # (B, N, 3)
                pred_batch_gw = endpoints["grasp_width_one_hot"]  # (B, N, 10)
                pred_batch_a = endpoints["approach_dir"]  # (B, N, 3)
                pred_batch_b = endpoints["baseline_dir"]  # (B, N, 3)

                # =============== Chamfer loss (start) ===============
                # Let's first get the correspondences and pred-to-gt
                nn_for_pred = knn_points(pred_batch_xyz, batch_xyz)
                nn_for_gt = knn_points(batch_xyz, pred_batch_xyz)

                # Chamfer Loss
                dists_for_pred = nn_for_pred.dists[..., 0]  # (B, N)
                dists_for_gt = nn_for_gt.dists[..., 0]  # (B, N)
                chamfer_for_pred = dists_for_pred.sum(1) / pred_batch_xyz.size(
                    1
                )  # (B,)
                chamfer_for_gt = dists_for_gt.sum(1) / batch_xyz.size(1)  # (B)
                loss_chamfer = chamfer_for_pred + chamfer_for_gt  # (B)
                loss_chamfer_reduced = loss_chamfer.sum() / loss_chamfer.size(
                    0
                )  # scalar
                # =============== Chamfer loss (end) ===============

                # =============== Grasp loss (start) ===============
                # Find points-clouds who have low chamfer scores:
                low_chamfer_batch_indices = torch.where(
                    loss_chamfer < MIN_CHAMFER_LOSS_FOR_GRASP_EST
                )[
                    0
                ]  # shape is like a single dimensional array (m,)
                if len(low_chamfer_batch_indices) == 0:
                    loss_grasp_succes = 0
                    loss_grasp_width = 0
                    loss_grasp_ori = 0
                else:
                    # - choose gs for pcls with low chamfer loss
                    pred_batch_gs = endpoints["success"]
                    pred_batch_gs_low_chamfer = pred_batch_gs[low_chamfer_batch_indices]
                    gt_batch_gs = batch_target["success"]
                    gt_batch_gs_low_chamfer = gt_batch_gs[low_chamfer_batch_indices]

                    # -------- Grasp-success-loss for pred pointcloud
                    # - Find indices for pred-pcl whose distance from nn is lower than a threshold
                    dists_for_pred = dists_for_pred[
                        low_chamfer_batch_indices
                    ]  # (B', N)
                    batch_indices, point_indices = torch.where(
                        dists_for_pred < PROPAGATION_RADIUS
                    )
                    # - Compute predicted-batch-gs for only these indices
                    final_pred_batch_gs = pred_batch_gs_low_chamfer[
                        batch_indices, point_indices
                    ]
                    # - For finding the corresponding gt-grasp-success:
                    #   * First use the batch_indices, point_indices to find nn indices in gt-pcl
                    nn_ind_for_pred = nn_for_pred.idx[..., 0][
                        low_chamfer_batch_indices
                    ]  # (B', N)
                    nn_batch_indices_for_pred = batch_indices
                    nn_point_indices_for_pred = nn_ind_for_pred[
                        batch_indices, point_indices
                    ]
                    #   * Use the gt-pcl indices to compute corresponding grasp-success
                    final_gt_batch_gs = gt_batch_gs_low_chamfer[
                        nn_batch_indices_for_pred, nn_point_indices_for_pred
                    ]
                    loss_for_pred_nn_grasp_success = (
                        torch.nn.functional.binary_cross_entropy(
                            final_pred_batch_gs, final_gt_batch_gs, reduction="none"
                        )
                    )

                    # Find batch-boundaries so that we can choose topk-errors for a
                    #  given point-cloud in batch
                    queries = torch.arange(1, args.batch_size).cuda()
                    batch_boundaries = torch.searchsorted(batch_indices, queries)
                    batch_boundaries = torch.cat(
                        (
                            torch.tensor([0]).cuda(),
                            batch_boundaries,
                            torch.tensor([len(batch_indices)]).cuda(),
                        ),
                        dim=0,
                    )
                    loss_gs_batched = [
                        loss_for_pred_nn_grasp_success[b_start:b_end]
                        for b_start, b_end in zip(
                            batch_boundaries[:-1], batch_boundaries[1:]
                        )
                        if b_start != b_end
                    ]
                    loss_gs_batched_topk = [
                        torch.topk(
                            x, k=min(len(x), GRASP_LOSS_TOP_K), sorted=False
                        ).values.mean()
                        for x in loss_gs_batched
                    ]
                    topk_loss_for_pred_nn_grasp_success = 0
                    for x in loss_gs_batched_topk:
                        topk_loss_for_pred_nn_grasp_success += x
                    if len(loss_gs_batched_topk) != 0:
                        topk_loss_for_pred_nn_grasp_success /= len(loss_gs_batched_topk)

                    # -------- Grasp-success-loss for gt pointcloud
                    # - Find indices for gt-pcl whose distance from nn is lower than threshold
                    dists_for_gt = dists_for_gt[low_chamfer_batch_indices]
                    batch_indices, point_indices = torch.where(
                        dists_for_gt < PROPAGATION_RADIUS
                    )
                    # - Computer gt-batch-gs for these indices
                    final_gt_batch_gs = gt_batch_gs_low_chamfer[
                        batch_indices, point_indices
                    ]
                    # - For finding the corresponding pred-grasp-success:
                    #   * First use the batch_indices, point_indices to find nn indices in pred-pcl
                    nn_ind_for_gt = nn_for_gt.idx[..., 0][
                        low_chamfer_batch_indices
                    ]  # (B', N)
                    nn_batch_indices_for_gt = batch_indices
                    nn_point_indices_for_gt = nn_ind_for_gt[
                        batch_indices, point_indices
                    ]
                    #   * Use these indices to compute corresponding pred-grasp-success
                    final_pred_batch_gs = pred_batch_gs_low_chamfer[
                        nn_batch_indices_for_gt, nn_point_indices_for_gt
                    ]
                    loss_for_gt_nn_grasp_success = (
                        torch.nn.functional.binary_cross_entropy(
                            final_pred_batch_gs, final_gt_batch_gs, reduction="none"
                        )
                    )
                    # - Keep only tok-k values with largest loss
                    topk_loss_for_gt_nn_grasp_success = torch.topk(
                        loss_for_gt_nn_grasp_success, k=GRASP_LOSS_TOP_K, sorted=False
                    ).values.mean()

                    loss_grasp_succes = (
                        topk_loss_for_pred_nn_grasp_success
                        + topk_loss_for_gt_nn_grasp_success
                    ) / 2

                    # =============== Grasp loss (end)===============

                    if not args.skip_grasp_width_loss or not args.skip_grasp_ori_loss:
                        # ================== Grasp Width Loss =====================
                        # - Step 1: Just consider low chamfer point-clouds in batch.
                        gt_batch_xyz_low_chamfer = batch_xyz[low_chamfer_batch_indices]
                        # - Step 2: Create Hetrogeneous-batch P+ point-cloud
                        #   * For every point-cloud in batch, first estimate the success
                        #     indices and max number of points in any of these point-clouds
                        low_chamfer_batch_size = len(low_chamfer_batch_indices)
                        batch_success_indices_array = torch.empty(
                            (low_chamfer_batch_size, batch_xyz.size(1)),
                            dtype=torch.long,
                        ).cuda()
                        batch_num_success_indices = torch.empty(
                            low_chamfer_batch_size, dtype=torch.int
                        ).cuda()
                        max_num_points = 0
                        for batch_ind in range(low_chamfer_batch_size):
                            point_success_indices = torch.where(
                                gt_batch_gs_low_chamfer[batch_ind]
                            )[0]
                            point_success_indices_length = len(point_success_indices)
                            max_num_points = max(
                                max_num_points, point_success_indices_length
                            )
                            batch_success_indices_array[
                                batch_ind, :point_success_indices_length
                            ] = point_success_indices
                            batch_num_success_indices[
                                batch_ind
                            ] = point_success_indices_length

                        #   * Then create the batched point-clouds of heterogeneous size
                        batch_gt_succ_xyz = (
                            torch.empty((low_chamfer_batch_size, max_num_points, 3))
                            .cuda()
                            .to(torch.float)
                        )  # (B', N', 3)
                        batch_gt_succ_xyz_lengths = (
                            torch.empty(low_chamfer_batch_size).cuda().to(torch.long)
                        )
                        for batch_ind in range(low_chamfer_batch_size):
                            num_succ_indices = batch_num_success_indices[batch_ind]
                            succ_indices = batch_success_indices_array[
                                batch_ind, :num_succ_indices
                            ]
                            len_pcl = succ_indices.size(0)
                            batch_gt_succ_xyz_lengths[batch_ind] = len_pcl
                            batch_gt_succ_xyz[
                                batch_ind, :len_pcl, :
                            ] = gt_batch_xyz_low_chamfer[batch_ind][succ_indices]
                        # - Step 3: Find nearest neighbors for P+ in pred:
                        pred_batch_xyz_low_chamfer = pred_batch_xyz[
                            low_chamfer_batch_indices
                        ]  # (B', N)
                        nn_for_gt_succ = knn_points(
                            batch_gt_succ_xyz,
                            pred_batch_xyz_low_chamfer,
                            lengths1=batch_gt_succ_xyz_lengths,
                        )
                        # - Step 4: Filter out the neighbors that are far
                        gt_batch_gw_low_chamfer = gt_batch_gw[
                            low_chamfer_batch_indices
                        ]  # (B', N, 10)
                        (
                            is_pred_nearby_batch_indices,
                            is_pred_nearby_point_indices,
                        ) = torch.where(
                            nn_for_gt_succ.dists[..., 0] < PROPAGATION_RADIUS
                        )
                        # - Step 5: Estimating the gt-grasp-width vector: is_pred_nearby-indices will give
                        #   me indices in the hetero-gt-pcl. I will transform those indices back to the
                        #   low_chamfer_gt_point_cloud
                        #   * First, discard the batch, points which were heteregeneous to get correct-idx
                        #     in success-point-cloud
                        is_correct = (
                            is_pred_nearby_point_indices
                            < batch_gt_succ_xyz_lengths[is_pred_nearby_batch_indices]
                        )
                        batch_indices = is_pred_nearby_batch_indices[is_correct]
                        point_indices = is_pred_nearby_point_indices[is_correct]
                        # - Step 6: Estimating the pred-grasp-width-vector:
                        #   (batch_indices, point_indices) will give you indices in the hetero-gt-pcl
                        #       which was input to the nn computation
                        #   so we can use (batch_indices, point_indices) to locate points in low-chamfer-
                        #       pred-pcl which was the second input to the nn computation
                        point_indices_for_pred = nn_for_gt_succ.idx[..., 0][
                            batch_indices, point_indices
                        ]
                        point_indices_low_chamfer = batch_success_indices_array[
                            batch_indices, point_indices
                        ]

                    if not args.skip_grasp_width_loss:
                        #   * Use mapping from success-point-cloud indices to low-chamfer-pcl indices
                        target_gw = gt_batch_gw_low_chamfer[
                            batch_indices, point_indices_low_chamfer, :
                        ]

                        pred_batch_gw_low_chamfer = pred_batch_gw[
                            low_chamfer_batch_indices
                        ]
                        inp_gw = pred_batch_gw_low_chamfer[
                            batch_indices, point_indices_for_pred, :
                        ]

                        # Step 7: Compute losses
                        loss_grasp_width = torch.nn.functional.binary_cross_entropy(
                            inp_gw, target_gw, weight=GRASP_WIDTH_BIN_WEIGHTS
                        )
                    else:
                        loss_grasp_width = 0

                        # TODO: should the same thing be done for predpcl to gt pcl?
                        # ================== Grasp Width Loss (end) =====================

                        # ================== Grasp orientation loss (start) =============
                    if not args.skip_grasp_ori_loss:
                        # Step 1: filter out point-clouds with high chamfer loss
                        gt_batch_a_low_chamfer = gt_batch_a[
                            low_chamfer_batch_indices
                        ]  # (B', N, 3)
                        gt_batch_b_low_chamfer = gt_batch_b[
                            low_chamfer_batch_indices
                        ]  # (B', N, 3)
                        pred_batch_a_low_chamfer = pred_batch_a[
                            low_chamfer_batch_indices
                        ]  # (B', N, 3)
                        pred_batch_b_low_chamfer = pred_batch_b[
                            low_chamfer_batch_indices
                        ]  # (B', N, 3)
                        # Step 2: Use the indices in grasp-width calculations, i.e., points
                        # in low-chamfer-pcls which are close to the gt-points
                        pred_a = pred_batch_a_low_chamfer[
                            batch_indices, point_indices_for_pred, :
                        ]  # (K, 3)
                        pred_b = pred_batch_b_low_chamfer[
                            batch_indices, point_indices_for_pred, :
                        ]  # (K, 3)
                        pred_xyz = pred_batch_xyz_low_chamfer[
                            batch_indices, point_indices_for_pred, :
                        ]  # (K, 3)
                        pred_xyz_detached = pred_xyz.detach()

                        def get_keypoints_from_a_b_xyz_wv(a, b, xyz, wv, v):
                            """
                            Transform gripper-frame-keypoints for predictions
                            a: approach dir  (K, 3)
                            b: baseline dir  (K, 3)
                            xyz: points  (K, 3)
                            wv: grasp-width values (K)
                            v: gripper-control-poitns (5, 3)
                            returns: Transformed keypoints (K, 5, 3)
                            """
                            t = (
                                xyz + (wv / 2).unsqueeze(dim=-1) * b + GRIPPER_DEPTH * a
                            )  # (K, 3)
                            r = torch.empty((a.size(0), 3, 3)).cuda().to(torch.float)
                            r[:, :, 0] = b
                            r[:, :, 1] = torch.cross(a, b)
                            r[:, :, 2] = a
                            v_batch = v.unsqueeze(dim=0).expand(
                                r.size(0), -1, -1
                            )  # (K, 5, 3)
                            rt = r.transpose(1, 2)  # (K, 3, 3)
                            return torch.bmm(v_batch, rt) + t.unsqueeze(
                                dim=1
                            )  # (K, 5, 3)

                        gt_a = gt_batch_a_low_chamfer[
                            batch_indices, point_indices_low_chamfer, :
                        ]
                        gt_b = gt_batch_b_low_chamfer[
                            batch_indices, point_indices_low_chamfer, :
                        ]
                        gt_gwv = gt_batch_gwv[low_chamfer_batch_indices][
                            batch_indices, point_indices_low_chamfer
                        ]
                        # NOTE: Should I stop the gradients to flow in point-coordinates?
                        gt_keypoints = get_keypoints_from_a_b_xyz_wv(
                            gt_a,
                            gt_b,
                            pred_xyz_detached,
                            gt_gwv,
                            gripper_control_points,
                        )  # (K, 5, 3)
                        # NOTE: Should I stop the gradients to flow in point-coordinates?
                        gt_keypoints_sym = get_keypoints_from_a_b_xyz_wv(
                            gt_a,
                            gt_b,
                            pred_xyz_detached,
                            gt_gwv,
                            gripper_control_points_sym,
                        )  # (K, 5, 3)
                        # NOTE: It's not clear from contact-grasp-net paper that for
                        # 6DoF-Grasp-Loss, whether or not they use predicted widths. Here I
                        # chose to use gt-widths and only use predicted_a and predicted_b
                        # for this loss computation.
                        pred_keypoints_pos_b = get_keypoints_from_a_b_xyz_wv(
                            pred_a,
                            pred_b,
                            pred_xyz_detached,
                            gt_gwv,
                            gripper_control_points,
                        )  # (K, 5, 3)
                        dist_pos = torch.linalg.norm(
                            gt_keypoints - pred_keypoints_pos_b, dim=-1
                        ).mean(
                            dim=-1
                        )  # (K)
                        dist_sym_pos = torch.linalg.norm(
                            gt_keypoints_sym - pred_keypoints_pos_b, dim=-1
                        ).mean(
                            dim=-1
                        )  # (K)
                        dists_to_consider = [dist_pos, dist_sym_pos]

                        if not args.treat_flippped_bd_as_bad:
                            pred_keypoints_neg_b = get_keypoints_from_a_b_xyz_wv(
                                pred_a,
                                -1 * pred_b,
                                pred_xyz_detached,
                                gt_gwv,
                                gripper_control_points,
                            )  # (K, 5, 3)
                            dist_neg = torch.linalg.norm(
                                gt_keypoints - pred_keypoints_neg_b, dim=-1
                            ).mean(
                                dim=-1
                            )  # (K)
                            dist_sym_neg = torch.linalg.norm(
                                gt_keypoints_sym - pred_keypoints_neg_b, dim=-1
                            ).mean(
                                dim=-1
                            )  # (K)
                            dists_to_consider += [dist_neg, dist_sym_neg]

                        dists = torch.stack(dists_to_consider, dim=0)
                        try:
                            dist_min = torch.min(dists, dim=0).values
                            loss_grasp_ori = (
                                pred_batch_gs_low_chamfer[
                                    batch_indices, point_indices_for_pred
                                ]
                                * dist_min
                            ).mean()  # (scalar)
                        except Exception as e:
                            print("Exception occurred in loss grasp ori")
                            print(e)
                            print(dists.shape)
                            loss_grasp_ori = 0
                    else:
                        loss_grasp_ori = 0
                    # ================== Grasp orientation loss (end) =============
                loss_grasp_parameters = (
                    WEIGHT_GRASP_SUCCESS_LOSS * loss_grasp_succes
                    + WEIGHT_GRASP_WIDTH_LOSS * loss_grasp_width
                    + WEIGHT_GRASP_ORI_LOSS * loss_grasp_ori
                )
                loss = (
                    WEIGHT_SHAPE_LOSS * loss_chamfer_reduced
                    + WEIGHT_GRASP_PARAMETERS * loss_grasp_parameters
                )
                loss.backward()
                optimizer.step()
                global_step += 1
                epoch_batch_idx += 1

                # write results to tensorboard
                # tb_writer.add_summary(summary, global_step)
                if epoch_batch_idx % 10 == 0:
                    logger.info(
                        f"Batch: {epoch_batch_idx} LOSS-{loss:.4f}"
                        f" CH-{loss_chamfer_reduced:.4f} GS-{loss_grasp_succes:.4f}"
                        f" GW-{loss_grasp_width:.4f}"
                        f" GO-{loss_grasp_ori:.4f}"
                    )
                    # logger.info(f"Batch {epoch_batch_idx} Loss:{loss.item()}")

                if epoch_batch_idx % args.save_every_n_batch == 0:
                    # save model after each epoch
                    torch.save(
                        estimator.state_dict(),
                        logs_dir / f"model_epoch_{epoch}_batch_{epoch_batch_idx}.pth",
                    )
        # evaluate one epoch
        logger.info(
            "Time {0}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))
                + ", "
                + "Epoch %02d" % epoch
                + ", "
                + "Testing started"
            )
        )
        estimator.eval()
        val_loss = 0.0
        for i, data in enumerate(val_dataloader, 1):
            inp, batch_target = data
            batch_xyz = inp["xyz"]
            batch_scales = inp["scale"]
            batch_xyz = batch_xyz[:, :, :3].cuda().to(torch.float)
            batch_scales = batch_scales.cuda().to(torch.float)
            batch_target = {
                key: value.cuda().to(torch.float) for key, value in batch_target.items()
            }
            _, endpoints = estimator(batch_xyz, batch_scales)
            point_cloud = endpoints["xyz"]
            chamfer_loss, _ = chamfer_distance(point_cloud, batch_xyz)
            val_loss += chamfer_loss.item()
            logger.info("Batch {0} Loss:{1:f}".format(i, loss))
        val_loss = val_loss / i
        # summary = tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=val_loss)])
        # tb_writer.add_summary(summary, global_step)
        logger.info("Epoch {0:02d} test average loss: {1:06f}".format(epoch, val_loss))
        logger.info(
            ">>>>>>>>----------Epoch {:02d} test finish---------<<<<<<<<".format(epoch)
        )
        torch.save(
            estimator.state_dict(),
            logs_dir / f"model_epoch_{epoch}_batch_{epoch_batch_idx}.pth",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_point",
        type=int,
        default=get_n_points(),
        help="number of points in the input point-cloud",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=get_emb_dim(),
        help="dimension of latent embedding",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="scale-shape-grasp dataset root",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=get_num_cpus(),
        help="number of data loading workers",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument(
        "--lr",
        type=float,
        default=get_shape_grasp_ae_initial_lr(),
        help="initial learning rate",
    )
    parser.add_argument(
        "--start_epoch", type=int, default=1, help="which epoch to start"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=30, help="max number of epochs to train"
    )
    parser.add_argument(
        "--resume_model",
        type=str,
        default=None,
        help="If passed, the model will be initialized from this saved model path",
    )
    parser.add_argument(
        "--repeat_epoch",
        type=int,
        default=30,
        help="Every epoch is repeated this many times before testing and lr decay start",
    )
    parser.add_argument(
        "--category_id",
        type=str,
        default=None,
        help="If passed, the network will only be trained for this category",
    )
    parser.add_argument(
        "--logs_dir_suffix",
        type=str,
        default=None,
        help="short string suffix to identify the experiment directory",
    )
    parser.add_argument(
        "--min_chamfer_loss_for_grasp_losses",
        type=float,
        default=get_min_chamfer_loss_for_grasp_losses(),
    )
    parser.add_argument(
        "--grasp_loss_topk",
        type=int,
        default=get_grasp_loss_topk(),
    )
    parser.add_argument(
        "--weight_shape_loss",
        type=float,
        default=get_weight_shape_loss(),
    )
    parser.add_argument("--save_every_n_batch", type=int, default=5000)
    parser.add_argument("--skip_grasp_width_loss", action="store_true")
    parser.add_argument("--skip_grasp_ori_loss", action="store_true")
    # --treat_flipped_bd_as_bad
    # For grasp orientation loss computation, we find the nearest neighbor point of the
    #  predicted point in the ground truth point-cloud. A lot of ground-truth
    #  meshes in the NOCS dataset have very thin surfaces (example: laptop, bowl etc.).
    #  For such meshes, a predicted point can either get associated with either the
    #  inner surface or the outer surface of the ground truth point-cloud. Based on the
    #  surface association, the sign of the baseline direction will change.
    #  Due to this reason, not ignoring the sign of baseline-direction prediction makes
    #  training harder. Note that the sign of the baseline direction can be easily
    #  computed during inference time using the predicted geometry.
    parser.add_argument(
        "--treat_flippped_bd_as_bad",
        action="store_true",
        help="""
        By default, for computing loss of baseline direction, we ignore its sign.
        If this flag is passed, the sign is not ignored and the baseline direction
        prediction will be penalized heavily for incorrect sign prediction
        """,
    )
    parser.add_argument(
        "--decay_step",
        type=int,
        default=int(1e5),
        help="decay learning rate after this many number of batches",
    )
    parser.add_argument(
        "--choose_bd_sign",
        type=bool,
        default=False,
        help="""
        If passed, decide the sign of baseline-direction analytically based on
        the predicted point-cloud
        """,
    )
    args = parser.parse_args()
    args.decay_rate = [1.0, 0.6, 0.3, 0.1]

    train_net(args)


if __name__ == "__main__":
    main()
