from typing import Union, Optional, Any, Dict
import numpy as np
from l5kit.geometry import transform_points
import torch

__all__ = [
    'traj_stat', 'classify_traj', 'comp_val', 'filter_traj'
]


def trajectory_stat(
        history_positions: np.array,
        target_positions: np.array,
        centroid: np.array,
        world_to_image: np.array,
) -> Any:
    history_pixels = transform_points(history_positions + centroid, world_to_image)
    history_pixels -= history_pixels[0]
    history_y_change = history_pixels[np.argmax(np.abs(history_pixels[:, 1])), 1]
    history_x_change = history_pixels[np.argmax(np.abs(history_pixels[:, 0])), 0]

    target_pixels = transform_points(target_positions + centroid, world_to_image)
    target_pixels -= target_pixels[0]
    target_y_change = target_pixels[np.argmax(np.abs(target_pixels[:, 1])), 1]
    target_x_change = target_pixels[np.argmax(np.abs(target_pixels[:, 0])), 0]

    hist_diff = np.linalg.norm(np.diff(history_positions, axis=0), axis=1)
    history_speed = hist_diff.sum() / history_positions.shape[0]
    history_acceleration = (hist_diff[-1] - hist_diff[0]) / hist_diff.shape[0]

    target_diff = np.linalg.norm(np.diff(target_positions, axis=0), axis=1)
    target_speed = target_diff.sum() / target_positions.shape[0]
    target_acceleration = (target_diff[-1] - target_diff[0]) / target_diff.shape[0]

    total_acceleration = (target_diff[-1] - hist_diff[0]) / (target_diff.shape[0] + hist_diff.shape[0])

    return ('history_y_change', history_y_change), ('history_x_change', history_x_change), \
           ('target_y_change', target_y_change), ('target_x_change', target_x_change), \
           ('history_speed', history_speed), ('history_acceleration', history_acceleration), \
           ('target_speed', target_speed), ('target_acceleration', target_acceleration), \
           ('total_acceleration', total_acceleration)


def traj_stat(traj: dict, predicted_targets=None) -> Any:
    targets = predicted_targets if predicted_targets is not None else traj['target_positions']
    return trajectory_stat(traj['history_positions'], targets,
                           traj['centroid'], traj['world_to_image'])


def classify_traj(
        hist_y_change: np.array,
        tar_y_change: np.array,
        speed_change: np.array,
        turn_thresh: Optional[float] = 3.,
        speed_thresh: Optional[float] = 0.5,
        prefix: Optional[Any] = '',
        matrix: Optional[bool] = False
) -> Union[tuple, str]:
    if np.abs(tar_y_change) > turn_thresh:
        target = 'D' if tar_y_change < 0. else 'U'
    else:
        target = 'N'
    if np.abs(hist_y_change) > turn_thresh:
        history = 'U' if hist_y_change < 0. else 'D'
    else:
        history = 'N'

    if np.abs(speed_change) > speed_thresh:
        speed = 'D' if speed_change < 0. else 'U'
    else:
        speed = 'N'
    if matrix:
        conv = lambda x: 1 if x == 'N' else 0 if x == 'U' else 2
        return conv(history), conv(target), conv(speed)
    return f'{prefix}{history}{target}{speed}'


def comp_val(hist_change, tar_change, speed_change, traj_cls: str):
    if traj_cls[1] == 'N':
        return abs(hist_change), abs(speed_change)
    elif traj_cls[0] == 'N':
        return abs(tar_change), abs(speed_change)
    return abs(tar_change) + abs(hist_change), abs(speed_change)


def filter_traj(traj: dict, static_hist_thresh: Optional[float] = 1.):
    value = traj['target_availabilities'].sum()
    if value != traj['target_availabilities'].shape[0]:
        return 'target', value
    value = traj['history_availabilities'].sum()
    if value != traj['history_availabilities'].shape[0]:
        return 'history', value
    value = np.linalg.norm(np.diff(traj['history_positions'], axis=0), axis=1).sum()
    if static_hist_thresh and value < static_hist_thresh:
        return 'static', value  # filter scenes with static history
    return False
