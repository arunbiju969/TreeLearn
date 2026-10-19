import geopandas
import alphashape
import numpy as np
import pandas as pd
import pickle
import os
import os.path as osp
import tqdm
import torch
import random
import laspy
import shapely
from shapely.geometry import Point, Polygon
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN
import hdbscan
from tree_learn.util.data_preparation import voxelize, compute_features, load_data, SampleGenerator


N_JOBS = 10 # number of threads/processes to use for several functions that have multiprocessing/multithreading enabled



# function to generate tiles
def generate_tiles(cfg, forest_path, logger, return_type='voxelized'):
    plot_name = os.path.basename(forest_path)[:-4]
    base_dir = os.path.dirname(os.path.dirname(forest_path))

    # dirs for data saving
    voxelized_dir = osp.join(base_dir, f'forest_voxelized{cfg.voxel_size}')
    features_dir = osp.join(base_dir, 'features')
    save_dir = osp.join(base_dir, 'tiles')
    os.makedirs(voxelized_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # voxelize forest and optionally calculate hash mapping to original points
    logger.info('voxelizing forest...')
    save_path_voxelized = osp.join(voxelized_dir, f'{plot_name}.npz')
    save_path_voxelized_original_idx = osp.join(voxelized_dir, f'{plot_name}_original_idx.pkl')
    save_path_hash_mapping = osp.join(voxelized_dir, f'{plot_name}_hash_mapping.pkl')
    if (not osp.exists(save_path_voxelized)) or (return_type == 'original' and not osp.exists(save_path_voxelized_original_idx)):
        data = load_data(forest_path)
        data, original_idx = voxelize(data, cfg.voxel_size)
        data = data.astype(np.float32)
        data = np.round(data, 2)
        np.savez_compressed(save_path_voxelized, points=data[:, :3], labels=data[:, 3])

        if return_type == 'original':
            original_idx = [list(item) for item in original_idx]
            # hash mapping
            hash_values = get_hash_values(data[:, :3])
            hash_mapping = get_hash_mapping(hash_values, original_idx)
            with open(save_path_voxelized_original_idx, 'wb') as f:
                pickle.dump(original_idx, f)
            with open(save_path_hash_mapping, 'wb') as f:
                pickle.dump(hash_mapping, f)
            del hash_values, original_idx, hash_mapping
            
    # calculating features
    logger.info('calculating features...')
    save_path_features = osp.join(features_dir, f'{plot_name}.npz')
    if not osp.exists(save_path_features):
        data = load_data(save_path_voxelized)
        features = compute_features(points=data[:, :3].astype(np.float64), search_radius=cfg.search_radius_features, feature_names=['verticality'], num_threads=N_JOBS)
        np.savez_compressed(save_path_features, features=features)
        
    # add cfg args that were generated dynamically based on plot
    logger.info('getting tiles...')
    cfg.sample_generator.plot_path = osp.join(voxelized_dir, f'{plot_name}.npz')
    cfg.sample_generator.features_path = osp.join(features_dir, f'{plot_name}.npz')
    cfg.sample_generator.save_dir = save_dir

    # generate tiles
    obj = SampleGenerator(**cfg.sample_generator)
    obj.tile_generate_and_save(cfg.inner_edge, cfg.outer_edge, cfg.stride, logger=logger)


# get offset and semantic predictions for all tiles
def get_pointwise_preds(model, dataloader, config, logger=None, save_backbone_feats=False):
    with torch.no_grad():
        model.eval()
        semantic_prediction_logits, offset_predictions, semantic_labels, offset_labels, coords, instance_labels, input_feats = [], [], [], [], [], [], []
        backbone_feats = [] if save_backbone_feats else None
        for batch in tqdm.tqdm(dataloader):
            # get voxel_sizes to use in forward
            batch['voxel_size'] = config.voxel_size
            # forward
            try:
                output = model(batch, return_loss=False)
                offset_prediction, semantic_prediction_logit = output['offset_predictions'].cpu(), output['semantic_prediction_logits'].cpu()
                if save_backbone_feats:
                    backbone_feat = output['backbone_feats'].cpu()
            except Exception as e:
                if "reach zero!!!" in str(e):
                    if logger:
                        logger.info('Error in forward pass due to axis size collapse to zero during contraction of U-Net. If this does not happen too often, the results should not be influenced.')
                    continue 
                else:
                    raise

            batch['coords'] = batch['coords'] + batch['centers']
            mask_inner = batch['masks_inner']
            input_feats.append(batch['input_feats'][mask_inner])
            semantic_prediction_logits.append(semantic_prediction_logit[mask_inner])
            semantic_labels.append(batch['semantic_labels'][mask_inner])
            offset_predictions.append(offset_prediction[mask_inner])
            offset_labels.append(batch['offset_labels'][mask_inner])
            coords.append(batch['coords'][mask_inner])
            instance_labels.append(batch['instance_labels'][mask_inner])
            if save_backbone_feats:
                backbone_feats.append(backbone_feat[mask_inner])

    input_feats = torch.cat(input_feats, 0).numpy()
    semantic_prediction_logits, semantic_labels = torch.cat(semantic_prediction_logits, 0).numpy(), torch.cat(semantic_labels, 0).numpy()
    offset_predictions, offset_labels = torch.cat(offset_predictions, 0).numpy(), torch.cat(offset_labels, 0).numpy()
    coords, instance_labels = torch.cat(coords, 0).numpy(), torch.cat(instance_labels).numpy()
    if save_backbone_feats and backbone_feats is not None:
        backbone_feats = torch.cat(backbone_feats, 0).numpy()
    else:
        backbone_feats = None
    return semantic_prediction_logits, semantic_labels, offset_predictions, offset_labels, coords, instance_labels, backbone_feats, input_feats


# ensemble pointwise predictions of overlapping tiles to get a single offset and semantic prediction for each point
def ensemble(coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats, input_feats):
    n_feats = feats.shape[1] if feats is not None else 0
    n_input_feats = input_feats.shape[1]
    feats_col_names = [f'feats{i}' for i in range(n_feats)] if feats is not None else []
    input_feats_col_names = [f'input_feats{i}' for i in range(n_input_feats)]

    # Build a single combined float32 array instead of constructing many DataFrames and
    # concatenating them (which creates a full extra copy of all data and can exhaust RAM
    # on large point clouds). Column order:
    # [x, y, z, sem_scores1, sem_scores2, semantic_labels, offset_pred1..3,
    #  offset_lab1..3, instance_labels, feats..., input_feats...]
    n_points = coords.shape[0]
    n_cols = 3 + 2 + 1 + 3 + 3 + 1 + n_feats + n_input_feats
    combined = np.empty((n_points, n_cols), dtype=np.float32)
    combined[:, 0:3] = coords
    combined[:, 3:5] = semantic_scores
    combined[:, 5] = semantic_labels.reshape(-1)
    combined[:, 6:9] = offset_predictions
    combined[:, 9:12] = offset_labels
    combined[:, 12] = instance_labels.reshape(-1)
    if feats is not None:
        combined[:, 13:13 + n_feats] = feats
        combined[:, 13 + n_feats:] = input_feats
    else:
        combined[:, 13:] = input_feats

    # free the large inputs as soon as they are copied to reduce peak memory
    del coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats, input_feats

    col_names = (['x', 'y', 'z', 'sem_scores1', 'sem_scores2', 'semantic_labels',
                  'offset_pred1', 'offset_pred2', 'offset_pred3',
                  'offset_lab1', 'offset_lab2', 'offset_lab3', 'instance_labels']
                 + feats_col_names + input_feats_col_names)
    df = pd.DataFrame(combined, columns=col_names, copy=False)
    del combined

    # round coordinates in place (avoids a full copy of the coordinate columns)
    df['x'] = df['x'].round(2)
    df['y'] = df['y'].round(2)
    df['z'] = df['z'].round(2)

    grouped = df.groupby(['x', 'y', 'z']).mean().reset_index()
    del df

    # Convert columns to desired data types
    coords = grouped[['x', 'y', 'z']].to_numpy().astype('float32')
    semantic_scores = grouped[['sem_scores1', 'sem_scores2']].to_numpy().astype('float32')
    semantic_labels = grouped[['semantic_labels']].to_numpy().astype('int64').flatten()
    offset_predictions = grouped[['offset_pred1', 'offset_pred2', 'offset_pred3']].to_numpy().astype('float32')
    offset_labels = grouped[['offset_lab1', 'offset_lab2', 'offset_lab3']].to_numpy().astype('float32')
    instance_labels = grouped[['instance_labels']].to_numpy().astype('int64').flatten()
    feats = grouped[feats_col_names].to_numpy().astype('float32') if n_feats > 0 else None
    input_feats = grouped[input_feats_col_names].to_numpy().astype('float32')
    return coords, semantic_scores, semantic_labels, offset_predictions, offset_labels, instance_labels, feats, input_feats


# get tree predictions for all points by using DBSCAN.
def get_instances(coords, offset, semantic_prediction_logits, grouping_cfg, verticality_feat, tree_class_in_dataset, non_trees_label_in_grouping, not_assigned_label_in_grouping, start_num_preds):
    cluster_coords = coords + offset
    cluster_coords = cluster_coords[:, :3]

    # get tree coords whose offset magnitude and verticality feature is appropriate
    semantic_prediction_probs = torch.from_numpy(semantic_prediction_logits).float().softmax(dim=-1)
    tree_mask = semantic_prediction_probs[:, tree_class_in_dataset] >= grouping_cfg.tree_conf_thresh
    vertical_mask = verticality_feat > grouping_cfg.tau_vert
    offset_mask = np.abs(offset[:, 2]) < grouping_cfg.tau_off
    mask_cluster = tree_mask.numpy() & vertical_mask & offset_mask
    ind_cluster = np.where(mask_cluster)[0]
    cluster_coords_filtered = cluster_coords[ind_cluster]
    cluster_coords_filtered = cluster_coords_filtered[:, :2]
    
    # get predictions
    predictions = non_trees_label_in_grouping * np.ones(len(cluster_coords))
    predictions[tree_mask] = not_assigned_label_in_grouping

    # get predicted instances
    if grouping_cfg.use_hdbscan:
        pred_instances = group_hdbscan(cluster_coords_filtered, grouping_cfg.tau_min, not_assigned_label_in_grouping, start_num_preds)
    else:
        pred_instances = group_dbscan(cluster_coords_filtered, grouping_cfg.tau_group, grouping_cfg.tau_min, not_assigned_label_in_grouping, start_num_preds)
    predictions[ind_cluster] = pred_instances 
    return predictions.astype(np.int64)


# DBSCAN
def group_dbscan(cluster_coords, radius, npoint_thr, not_assigned_label_in_grouping, start_num_preds):
    clustering = DBSCAN(eps=radius, min_samples=2, n_jobs=N_JOBS).fit(cluster_coords)
    cluster_nums, n_points = np.unique(clustering.labels_, return_counts=True)
    valid_cluster_nums = cluster_nums[(n_points >= npoint_thr) & (cluster_nums != -1)]
    ind_valid = np.isin(clustering.labels_, valid_cluster_nums)
    clustering.labels_[ind_valid], _ = make_labels_consecutive(clustering.labels_[ind_valid], start_num=start_num_preds)
    clustering.labels_[np.logical_not(ind_valid)] = not_assigned_label_in_grouping
    return clustering.labels_


# HDBSCAN
def group_hdbscan(cluster_coords, npoint_thr, not_assigned_label_in_grouping, start_num_preds):
    clustering = hdbscan.HDBSCAN(min_cluster_size=npoint_thr, core_dist_n_jobs=4).fit(cluster_coords)
    cluster_nums, n_points = np.unique(clustering.labels_, return_counts=True)
    valid_cluster_nums = cluster_nums[(n_points >= npoint_thr) & (cluster_nums != -1)]
    ind_valid = np.isin(clustering.labels_, valid_cluster_nums)
    clustering.labels_[ind_valid], _ = make_labels_consecutive(clustering.labels_[ind_valid], start_num=start_num_preds)
    clustering.labels_[np.logical_not(ind_valid)] = not_assigned_label_in_grouping
    return clustering.labels_


# helper function that makes labels consecutive (e.g., 0, 1, 2, 3, ...); also returns back mapping dictionary to original labels
def make_labels_consecutive(labels, start_num):
    palette = np.unique(labels)
    palette = np.sort(palette)
    key = np.arange(0, len(palette))
    index = np.digitize(labels, palette, right=True)
    labels = key[index]
    labels = labels + start_num
    
    # Create a mapping dictionary from new values to original values
    mapping_dict = {new_label + start_num: original_label for new_label, original_label in enumerate(palette)}
    
    return labels, mapping_dict


# get all coordinates that are within a specified shape (e.g. an xy hull around a hull buffer). 
# This way, trees at the edge can be identified or a buffer can be removed at the edge of the plot
def get_coords_within_shape(coords, shape):
    if hasattr(shape, "geometry"):
        geom = shape.geometry.union_all() if hasattr(shape.geometry, "union_all") else shape.geometry.unary_union
    elif hasattr(shape, "union_all"):
        geom = shape.union_all()
    elif hasattr(shape, "unary_union"):
        geom = shape.unary_union
    else:
        geom = shape

    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    return shapely.contains_xy(geom, x, y)


# get a coarser grid of all xy points to calculate the hull
def grid_points(coords, grid_size):
    # Create a DataFrame from coordinates
    df = pd.DataFrame(coords, columns=['x', 'y'])
    
    # Assign each point to a grid cell by dividing coordinates by grid_size and flooring
    df['grid_x'] = (df['x'] // grid_size).astype(int)
    df['grid_y'] = (df['y'] // grid_size).astype(int)
    
    # Keep only one point per grid cell (e.g., the first occurrence)
    reduced_df = df.drop_duplicates(subset=['grid_x', 'grid_y'])
    
    # Return the reduced set of points as a numpy array
    return reduced_df[['x', 'y']].to_numpy()


# get buffer around hull
def get_hull_buffer(coords, alpha, buffersize):
    # create 2-dimensional hull of forest xy-coordinates
    coords_mean = np.mean(coords, axis=0, dtype=np.float64)
    coords = grid_points(coords - coords_mean, grid_size=0.25)
    
    # create 2-dimensional hull of forest xy-coordinates and from this create hull buffer
    hull_polygon = alphashape.alphashape(coords, alpha)
    hull_polygon = shift_hull(hull_polygon, coords_mean)
    hull_line = hull_polygon.boundary
    hull_line_geoseries = geopandas.GeoSeries(hull_line)
    hull_buffer = hull_line_geoseries.buffer(buffersize)
    hull_buffer_geodf = geopandas.GeoDataFrame(geometry=hull_buffer)
    return hull_buffer_geodf


# get hull
def get_hull(coords, alpha):
    # create 2-dimensional hull of forest xy-coordinates
    coords_mean = np.mean(coords, axis=0, dtype=np.float64)
    coords = grid_points(coords - coords_mean, grid_size=0.25)

    hull_polygon = alphashape.alphashape(coords, alpha)
    hull_polygon = shift_hull(hull_polygon, coords_mean)
    hull_polygon_geoseries = geopandas.GeoSeries(hull_polygon)
    hull_polygon_geodf = geopandas.GeoDataFrame(geometry=hull_polygon_geoseries)
    return hull_polygon_geodf


def shift_hull(hull_polygon, shift):
    assert isinstance(hull_polygon, Polygon), "failed to calculate concave hull. Set alpha=0 to use convex hull or set outer_remove=~"
    vertices = np.array(hull_polygon.exterior.coords)
    modified_vertices = vertices + shift
    hull_polygon = Polygon(modified_vertices)
    return hull_polygon
    
    
# get cluster means (used to identify edge trees)
def get_cluster_means(coords, labels):
    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df['label'] = labels
    cluster_means = df.groupby('label').mean().values 
    return cluster_means


# assign remaining tree points after initial clustering
def assign_remaining_points_nearest_neighbor(coords, predictions, remaining_points_idx, n_neighbors=5, chunk_size=1000000):
    predictions = np.copy(predictions)
    assert len(coords) == len(predictions) # input variable should be of same size
    query_idx = np.argwhere(predictions == remaining_points_idx).reshape(-1)
    reference_idx = np.argwhere(predictions != remaining_points_idx).reshape(-1)

    if len(query_idx) == 0 or len(reference_idx) == 0:
        return predictions.astype(np.int64)

    ref_coords = coords[reference_idx]
    ref_labels = predictions[reference_idx]
    query_coords = coords[query_idx]

    tree = cKDTree(ref_coords)
    query_preds = np.empty(len(query_coords), dtype=np.int64)

    for i in range(0, len(query_coords), chunk_size):
        chunk = query_coords[i:i + chunk_size]
        _, indices = tree.query(chunk, k=n_neighbors, workers=-1)
        chunk_labels = torch.from_numpy(ref_labels[indices])
        modes = torch.mode(chunk_labels, dim=1).values.numpy()
        query_preds[i:i + len(chunk)] = modes

    predictions[query_idx] = query_preds
    return predictions.astype(np.int64)


# propagate predictions from source to target coordinates (e.g. lower resolution to higher resolution; does not have an effect on the actual predictions)
def propagate_preds(source_coords, source_preds, target_coords, n_neighbors=5, n_jobs=-1, chunk_size=1000000):
    source_coords = source_coords.astype(np.float32)
    target_coords = target_coords.astype(np.float32)
    source_preds = source_preds.astype(np.int64)

    tree = cKDTree(source_coords)
    N = target_coords.shape[0]
    target_preds = np.empty(N, dtype=np.int64)

    for i in range(0, N, chunk_size):
        chunk = target_coords[i:i + chunk_size]
        _, indices = tree.query(chunk, k=n_neighbors, workers=-1)
        chunk_preds = torch.from_numpy(source_preds[indices])
        modes = torch.mode(chunk_preds, dim=1).values.numpy()
        target_preds[i:i + len(chunk)] = modes

    return target_preds


def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]


# save point clouds in different formats
def save_data(data, save_format, save_name, save_folder, use_offset=True):
    if save_format == "las" or save_format == "laz":
        # get points and labels
        assert data.shape[1] == 4
        points = data[:, :3]
        labels = data[:, 3].astype(np.int32)
        classification = np.ones_like(labels, dtype=np.uint8)
        classification[labels == 0] = 2 # terrain according to For-Instance labeling convention (https://zenodo.org/records/8287792)
        classification[labels != 0] = 4 # stem according to For-Instance labeling convention (https://zenodo.org/records/8287792)

        # Create a new LAS file
        header = laspy.LasHeader(version="1.2", point_format=3)
        if use_offset:
            mean_x, mean_y, mean_z = points.mean(0)
            header.offsets = [mean_x, mean_y, mean_z]
        else:
            header.offsets = [0, 0, 0]
        
        header.scales = [0.001, 0.001, 0.001]
        las = laspy.LasData(header)

        # Set the points and additional fields
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]

        las.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
        las.treeID = labels
        las.classification = classification

        # Generate a color for each unique label
        unique_labels = np.unique(labels)
        color_map = {label: generate_random_color() for label in unique_labels}

        # Assign colors based on label
        colors = np.array([color_map[label] for label in labels], dtype=np.uint16)
        colors[classification == 2] = [0, 0, 0]

        # Set RGB colors in the LAS file
        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

        # Write the LAS file to disk
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        las.write(save_path)
    elif save_format == "npy":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.save(save_path, data)
    elif save_format == "npz":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.savez_compressed(save_path, points=data[:, :3], labels=data[:, 3])
    elif save_format == "txt":
        save_path = osp.join(save_folder, f'{save_name}.{save_format}')
        np.savetxt(save_path, data)


# save individual tree point clouds
def save_treewise(coords, instance_preds, cluster_means_within_hull, insts_not_at_edge, save_format, plot_results_dir, non_trees_label_in_grouping):
    coords = coords - np.mean(coords, axis=0) # avoid large coordinates
    completely_inside_dir = os.path.join(plot_results_dir, 'completely_inside')
    trunk_base_inside_dir = os.path.join(plot_results_dir, 'trunk_base_inside')
    trunk_base_outside_dir = os.path.join(plot_results_dir, 'trunk_base_outside')
    os.makedirs(completely_inside_dir, exist_ok=True)
    os.makedirs(trunk_base_inside_dir, exist_ok=True)
    os.makedirs(trunk_base_outside_dir, exist_ok=True)

    unique_instances = np.unique(instance_preds)

    def _save_tree(i):
        pred_coord = coords[instance_preds == i]
        pred_coord = np.hstack([pred_coord, i * np.ones(len(pred_coord))[:, None]])
        if i == non_trees_label_in_grouping:
            # use offset=false here for easy visualization of all individual trees in cloudcompare
            save_data(pred_coord, save_format, 'non_trees', plot_results_dir, use_offset=False)
            return

        if cluster_means_within_hull[i-1] and insts_not_at_edge[i-1]:
            save_data(pred_coord, save_format, str(int(i)), completely_inside_dir, use_offset=False)
        elif cluster_means_within_hull[i-1] and not insts_not_at_edge[i-1]:
            save_data(pred_coord, save_format, str(int(i)), trunk_base_inside_dir, use_offset=False)
        elif not cluster_means_within_hull[i-1]:
            save_data(pred_coord, save_format, str(int(i)), trunk_base_outside_dir, use_offset=False)

    max_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_save_tree, unique_instances))


# get hash values of voxelized points
def get_hash_values(voxelized_points):
    hash_values = []
    for point in voxelized_points:
        point_tuple = tuple(point)
        hash_value = hash(point_tuple)
        hash_values.append(hash_value)
    return hash_values


# get for each hash value the original indices (indices of the original point cloud that each voxelized point corresponds to)
def get_hash_mapping(hash_values, original_idx):
    hash_mapping = {}
    for i, hash_value in enumerate(hash_values):
        hash_mapping[hash_value] = original_idx[i]
    return hash_mapping


# map voxelized points to original points
def propagate_preds_hash_full(coords, instance_preds, coords_to_return, hash_mapping):
    coords = np.round(coords, 2)
    hash_values = get_hash_values(coords)

    target_preds = np.empty(coords_to_return.shape[0], np.int64)
    not_yet_propagated = np.ones(coords_to_return.shape[0], bool) # in case that tile denoising took place, this is needed to propagate the remaining points. By default, denoising is not used anymore
    for i, hash_value in enumerate(hash_values):
        orig_indices = hash_mapping.get(hash_value)
        if orig_indices is not None:
            target_preds[orig_indices] = instance_preds[i]
            not_yet_propagated[orig_indices] = False

    return target_preds, not_yet_propagated


# map voxelized points to voxelized points (restores the original order of the voxelized points)
def propagate_preds_hash_vox(coords, instance_preds, coords_to_return):
    hash_values_original = np.array(get_hash_values(coords_to_return), np.int64)
    hash_values_current = np.array(get_hash_values(np.round(coords, 2)), np.int64)
    
    # Propagate predictions
    hash_to_pred_map = dict(zip(hash_values_current, instance_preds))  # Map hash values to predictions
    preds_to_return = np.array([hash_to_pred_map.get(h, -1) for h in hash_values_original])  # Use -1 for missing hashes

    # Separate valid and invalid mappings
    not_yet_propagated = preds_to_return == -1
    return preds_to_return, not_yet_propagated
