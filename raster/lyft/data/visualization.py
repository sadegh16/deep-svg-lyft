import torch
import numpy as np
from typing import Union, Optional, Any, Tuple
import matplotlib.pyplot as plt
from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory, PREDICTED_POINTS_COLOR
from captum.attr import Saliency, Occlusion
from captum.attr import visualization as viz


def _set_image_type(image: np.array):
    """
    Utility function to force the type of image pixels,
        A mysterious bug caused the torch.Tensor.numpy() to miss produce the resulting array's type
    :param image:
    :return:
    """
    image_np = image
    float_image = np.zeros(image_np.shape)
    for i, channel in enumerate(image_np):
        for j, row in enumerate(channel):
            for z, pixel in enumerate(row):
                float_image[i][j][z] = float(pixel)
    return float_image


def draw_single_image(
        rasterizer,
        image: np.array,
        centroid: np.array,
        world_to_image: np.array,
        target_positions: np.array,
        target_yaws: np.array,
        predicted_positions: Optional[np.array] = None,
        predicted_yaws: Optional[np.array] = None,
        target_color: Optional[tuple] = TARGET_POINTS_COLOR,
        predicted_color: Optional[tuple] = PREDICTED_POINTS_COLOR,
) -> torch.Tensor:
    """
    Produce a single RGB representation of the rasterized input image and its corresponding position prediction
    :param rasterizer:
    :param image:
    :param centroid:
    :param world_to_image:
    :param target_positions:
    :param target_yaws:
    :param predicted_positions:
    :param predicted_yaws:
    :param target_color:
    :param predicted_color:
    :return:
    """
    predicted_yaws = predicted_yaws if predicted_yaws is not None else target_yaws
    im = _set_image_type(rasterizer.to_rgb(image.cpu().data.numpy().transpose(1, 2, 0)))  # Todo enhance
    draw_trajectory(im, transform_points(
        target_positions.cpu().data.numpy() + centroid[:2].cpu().data.numpy(), world_to_image.cpu().data.numpy()),
                    target_yaws.cpu().data.numpy(), target_color)
    if predicted_positions is not None:
        draw_trajectory(im, transform_points(
            predicted_positions.cpu().data.numpy() + centroid[:2].cpu().data.numpy(),
            world_to_image.cpu().data.numpy()), predicted_yaws.cpu().data.numpy(), predicted_color)
    return np.uint8(im).transpose(2, 0, 1)


def draw_batch(rasterizer, batch: dict, outputs: Optional[torch.Tensor] = None,
               predicted_yaws: Optional[torch.Tensor] = None,
               target_color: Optional[tuple] = TARGET_POINTS_COLOR,
               predicted_color: Optional[tuple] = PREDICTED_POINTS_COLOR) -> np.array:
    """
    Creates a grid numpy array of RGB representation of the rasterized inputs and the corresponding
    predicted trajectory
    :param rasterizer: rasterizer used to create the batch
    :param batch: batch dictionary
    :param outputs:
    :param predicted_yaws:
    :param target_color: tuple of (r,g,b)
    :param predicted_color: tuple of (r,g,b)
    :return: a grid of rgb representation of the batch
    """
    return np.array([draw_single_image(
        rasterizer, image, batch['centroid'][i], batch['world_to_image'][i], batch['target_positions'][i],
        batch['target_yaws'][i], None if outputs is None else outputs[i],
        None if predicted_yaws is None else predicted_yaws[i], target_color, predicted_color) for i, image in
        enumerate(batch['image'])],
        dtype=np.uint8)


def saliency_map(
        batch: dict,
        saliency: Saliency,
        sign: str = 'all',
        method: str = 'blended_heat_map',
        use_pyplot: bool = False,
        fig_axis: tuple = None,
        mix_bg: bool = True,
        alpha_overlay: float = 0.7,
) -> Tuple[Any, torch.Tensor]:
    """
    :param batch: batch to visualise
    :param saliency: Saliency object initialised for trainer_module
    :param sign: sign of gradient attributes to visualise
    :param method: method of visualization to be used
    :param use_pyplot: whether to use pyplot
    :param mix_bg: whether to mix semantic/aerial map with vehicles
    :return: pair of figure and corresponding gradients tensor
    """

    batch['image'].requires_grad = True
    grads = saliency.attribute(batch['image'], abs=False, additional_forward_args=(
        batch['target_positions'], None if 'target_availabilities' not in batch else batch['target_availabilities'],
        False))
    batch['image'].requires_grad = False
    gradsm = grads.squeeze().cpu().detach().numpy()
    if len(gradsm.shape) == 3:
        gradsm = gradsm.reshape(1, *gradsm.shape)
    gradsm = np.transpose(gradsm, (0, 2, 3, 1))
    im = batch['image'].detach().cpu().numpy().transpose(0, 2, 3, 1)
    fig, axis = fig_axis if fig_axis is not None else plt.subplots(2 - mix_bg, im.shape[0], dpi=200, figsize=(6, 6))
    for b in range(im.shape[0]):
        if mix_bg:
            grad_norm = float(np.abs(gradsm[b, ...]).sum())
            viz.visualize_image_attr(
                gradsm[b, ...], im[b, ...], method=method,
                sign=sign, use_pyplot=use_pyplot,
                plt_fig_axis=(fig, axis if im.shape[0] == 1 else axis[b]),
                alpha_overlay=alpha_overlay,
                title=f'l1 grad: {grad_norm:.5f}',
            )
            ttl = (axis if im.shape[0] == 1 else axis[b]).title
            ttl.set_position([.5, 0.95])
            (axis if im.shape[0] == 1 else axis[b]).axis('off')
        else:
            for (s_channel, end_channel), row in [((im.shape[-1] - 3, im.shape[-1]), 0), ((0, im.shape[-1] - 3), 1)]:
                grad_norm = float(np.abs(gradsm[b, :, :, s_channel:end_channel]).sum())
                viz.visualize_image_attr(
                    gradsm[b, :, :, s_channel:end_channel], im[b, :, :, s_channel:end_channel], method=method,
                    sign=sign, use_pyplot=use_pyplot,
                    plt_fig_axis=(fig, axis[row] if im.shape[0] == 1 else axis[row][b]),
                    alpha_overlay=alpha_overlay, title=f'l1 grad: {grad_norm:.5f}',
                )
                ttl = (axis[row] if im.shape[0] == 1 else axis[row][b]).title
                ttl.set_position([.5, 0.95])
                (axis[row] if im.shape[0] == 1 else axis[row][b]).axis('off')
    return fig, grads


def draw_occlusion(
        batch: dict,
        occlusion: Occlusion,
        window: tuple = (4, 4),
        stride: tuple = (8, 8),
        sign: str = 'positive',
        method: str = 'blended_heat_map',
        use_pyplot: bool = False,
        outlier_perc: float = 2.,
        fig_axis: tuple = None,
        mix_bg: bool = True,
        alpha_overlay: float = 0.7
):
    """
    :param batch: batch to visualise
    :param occlusion: Occlusion object initialised for trainer_module
    :param window: Shape of patch (hyperrectangle) to occlude each input
    :param stride: step by which the occlusion hyperrectangle should be shifted by in each direction
    :param sign: sign of gradient attributes to visualise
    :param method: method of visualization to be used
    :param use_pyplot: whether to use pyplot
    :param mix_bg: whether to mix semantic/aerial map with vehicles
    :return: pair of figure and corresponding gradients tensor
    """

    strides = (batch['image'].shape[2] // stride[0], batch['image'].shape[3] // stride[1])
    window_size = (batch['image'].shape[2] // window[0], batch['image'].shape[3] // window[1])
    channels = batch['image'].shape[1]

    grads = occlusion.attribute(
        batch['image'], strides=(channels if mix_bg else channels - 3, *strides),
        sliding_window_shapes=(channels if mix_bg else channels - 3, *window_size),
        baselines=0,
        additional_forward_args=(
            batch['target_positions'],
            None if 'target_availabilities' not in batch else batch['target_availabilities'],
            False))

    gradsm = grads.squeeze().cpu().detach().numpy()
    if len(gradsm.shape) == 3:
        gradsm = gradsm.reshape(1, *gradsm.shape)
    gradsm = np.transpose(gradsm, (0, 2, 3, 1))
    im = batch['image'].detach().cpu().numpy().transpose(0, 2, 3, 1)
    fig, axis = fig_axis if fig_axis is not None else plt.subplots(2 - mix_bg, im.shape[0], dpi=200, figsize=(6, 6))
    for b in range(im.shape[0]):
        if mix_bg:
            viz.visualize_image_attr(
                gradsm[b, ...], im[b, ...], method=method,
                sign=sign, use_pyplot=use_pyplot,
                plt_fig_axis=(fig, axis if im.shape[0] == 1 else axis[b]),
                alpha_overlay=alpha_overlay, outlier_perc=outlier_perc,
            )
            ttl = (axis if im.shape[0] == 1 else axis[b]).title
            ttl.set_position([.5, 0.95])
            (axis if im.shape[0] == 1 else axis[b]).axis('off')
        else:
            for (s_channel, end_channel), row in [((im.shape[-1] - 3, im.shape[-1]), 0), ((0, im.shape[-1] - 3), 1)]:
                viz.visualize_image_attr(
                    gradsm[b, :, :, s_channel:end_channel], im[b, :, :, s_channel:end_channel], method=method,
                    sign=sign, use_pyplot=use_pyplot,
                    plt_fig_axis=(fig, axis[row] if im.shape[0] == 1 else axis[row][b]),
                    alpha_overlay=alpha_overlay, outlier_perc=outlier_perc,
                )
                ttl = (axis[row] if im.shape[0] == 1 else axis[row][b]).title
                ttl.set_position([.5, 0.95])
                (axis[row] if im.shape[0] == 1 else axis[row][b]).axis('off')
    return fig, grads


def visualize_batch(
        batch, rasterizer=None, output_positions=None, title: str = '', output_root: str = None,
        occlusion: Occlusion = None, saliency: Saliency = None, mix_bg=False,
        occlusion_options: dict = dict(), saliency_options: dict = dict(),
        wspace: float = 0.1, hspace: float = 0.1, unit_size: float = 1.5, title_size: int = 6, subtitle_size: int = 4,
        dpi: int = 200, alpha_overlay: float = 0.7
):
    prediction_position = None if 'outputs' not in batch else batch['outputs']
    prediction_position = output_positions if output_positions is not None else prediction_position

    if title_size is not None and title:
        plt.rc('axes', titlesize=subtitle_size)

    rows_count = sum(
        [rasterizer is not None, (occlusion is not None) * (2 - mix_bg), (saliency is not None) * (2 - mix_bg)])
    fig, rows = plt.subplots(
        rows_count, batch['image'].shape[0],
        figsize=((unit_size + wspace + hspace) * batch['image'].shape[0],
                 (unit_size + wspace + hspace) * 3 + (0.2 if title else 0)),
        dpi=dpi, gridspec_kw={'wspace': wspace, 'hspace': hspace}
    )
    i = 0
    if rasterizer is not None:
        res = draw_batch(rasterizer, batch, prediction_position)
        axis = rows if rows_count == 1 else rows[i]
        for idx, im in enumerate(res):
            (axis if len(res) == 1 else axis[idx]).imshow(im.transpose(1, 2, 0), aspect='equal')
            (axis if len(res) == 1 else axis[idx]).axis('off')
            if 'loss' in batch:
                (axis if len(res) == 1 else axis[idx]).set_title(f'loss: {batch["loss"][idx]}')
        i += 1
    if title:
        fig.suptitle(f'{title}', fontsize=title_size)

    if saliency is not None:
        axis = rows if rows_count == 1 else rows[i] if mix_bg else (rows[i], rows[i + 1])
        fig, _ = saliency_map(batch, saliency, use_pyplot=False, fig_axis=(fig, axis),
                              mix_bg=mix_bg, alpha_overlay=alpha_overlay, **saliency_options)
        i += 2 - mix_bg
    if occlusion is not None:
        axis = rows if rows_count == 1 else rows[i] if mix_bg else (rows[i], rows[i + 1])
        fig, _ = draw_occlusion(batch, occlusion, use_pyplot=False, fig_axis=(fig, axis),
                                mix_bg=mix_bg, alpha_overlay=alpha_overlay, **occlusion_options)
    if output_root:
        fig.savefig(f'{output_root}/{title}')
