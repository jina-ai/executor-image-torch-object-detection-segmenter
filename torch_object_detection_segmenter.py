__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"


from jina.logging.logger import JinaLogger
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
import torchvision.models.detection as detection_models

from jina import Document, DocumentArray, Executor, requests


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]

class TorchObjectDetectionSegmenter(Executor):
    """
    :class:`TorchObjectDetectionSegmenter` detects objects
    from an image using `torchvision detection models`
    and crops the images according tothe detected bounding boxes
    of the objects with a confidence higher than a threshold.
    :param model_name: the name of the model. Supported models include
        ``fasterrcnn_resnet50_fpn``, ``maskrcnn_resnet50_fpn`
    :param confidence_threshold: confidence value from which it
        considers a positive detection and therefore the object detected will be cropped and returned
    :param label_name_map: A Dict mapping from label index to label name, by default will be
        COCO_INSTANCE_CATEGORY_NAMES
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
        TODO: Allow changing the backbone
    """
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, model_name: Optional[str] = None,
                 on_gpu: bool = False,
                 default_traversal_paths: List[str] = ['r'],
                 default_batch_size: int = 32,
                 confidence_threshold: float = 0.0,
                 label_name_map: Optional[Dict[int, str]] = None,
                 *args, **kwargs):
        """Set constructor"""
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        self.on_gpu = on_gpu
        self.model_name = model_name or 'fasterrcnn_resnet50_fpn'
        self._default_channel_axis = 0
        self._default_batch_size = default_batch_size
        self._default_traversal_paths = default_traversal_paths
        self.confidence_threshold = confidence_threshold
        self.label_name_map = label_name_map or TorchObjectDetectionSegmenter.COCO_INSTANCE_CATEGORY_NAMES
        self.model = getattr(detection_models, self.model_name)(pretrained=True, pretrained_backbone=True).eval()

    def _predict(self, batch: List[np.ndarray]) -> 'np.ndarray':
        """
        Run the model for prediction
        :param img: the image from which to run a prediction
        :return: the boxes, scores and labels predictedx
        """
        import torch
        _input = torch.from_numpy(np.stack(batch).astype('float32'))

        if self.on_gpu:
            _input = _input.cuda()

        return self.model(_input)

    @requests
    def segment(self, docs: DocumentArray, parameters: dict, *args, **kwargs) -> List[Dict]:
        """
        Crop the input image array within DocumentArray.
        :param docs: docs containing the ndarrays of the images
        :return: a list of chunk dicts with the cropped images
        :param args:  Additional positional arguments e.g. traversal_paths, batch_size
        :param kwargs: Additional keyword arguments
        """

        def _move_channel_axis(img: 'np.ndarray', channel_axis_to_move: int,
                               target_channel_axis: int = -1) -> 'np.ndarray':
            """
            Ensure the color channel axis is the default axis.
            """
            if channel_axis_to_move == target_channel_axis:
                return img
            return np.moveaxis(img, channel_axis_to_move, target_channel_axis)

        def _load_image(blob: 'np.ndarray', channel_axis: int):
            """
            Load an image array and return a `PIL.Image` object.
            """

            from PIL import Image
            img = _move_channel_axis(blob, channel_axis)
            return Image.fromarray(img.astype('uint8'))

        def _crop_image(img, target_size: Union[Tuple[int, int], int], top: int = None, left: int = None,
                        how: str = 'precise'):
            """
            Crop the input :py:mod:`PIL` image.
            :param img: :py:mod:`PIL.Image`, the image to be resized
            :param target_size: desired output size. If size is a sequence like
                (h, w), the output size will be matched to this. If size is an int,
                the output will have the same height and width as the `target_size`.
            :param top: the vertical coordinate of the top left corner of the crop box.
            :param left: the horizontal coordinate of the top left corner of the crop box.
            :param how: the way of cropping. Valid values include `center`, `random`, and, `precise`. Default is `precise`.
                - `center`: crop the center part of the image
                - `random`: crop a random part of the image
                - `precise`: crop the part of the image specified by the crop box with the given ``top`` and ``left``.
                .. warning:: When `precise` is used, ``top`` and ``left`` must be fed valid value.
            """
            import PIL.Image as Image
            assert isinstance(img, Image.Image), 'img must be a PIL.Image'
            img_w, img_h = img.size
            if isinstance(target_size, int):
                target_h = target_w = target_size
            elif isinstance(target_size, Tuple) and len(target_size) == 2:
                target_h, target_w = target_size
            else:
                raise ValueError(f'target_size should be an integer or a tuple of two integers: {target_size}')
            w_beg = left
            h_beg = top
            if how == 'center':
                w_beg = int((img_w - target_w) / 2)
                h_beg = int((img_h - target_h) / 2)
            elif how == 'random':
                w_beg = np.random.randint(0, img_w - target_w + 1)
                h_beg = np.random.randint(0, img_h - target_h + 1)
            elif how == 'precise':
                assert (w_beg is not None and h_beg is not None)
                assert (0 <= w_beg <= (img_w - target_w)), f'left must be within [0, {img_w - target_w}]: {w_beg}'
                assert (0 <= h_beg <= (img_h - target_h)), f'top must be within [0, {img_h - target_h}]: {h_beg}'
            else:
                raise ValueError(f'unknown input how: {how}')
            if not isinstance(w_beg, int):
                raise ValueError(f'left must be int number between 0 and {img_w}: {left}')
            if not isinstance(h_beg, int):
                raise ValueError(f'top must be int number between 0 and {img_h}: {top}')
            w_end = w_beg + target_w
            h_end = h_beg + target_h
            img = img.crop((w_beg, h_beg, w_end, h_end))
            return img, h_beg, w_beg

        def _get_input_data(docs: DocumentArray, parameters: dict):
            traversal_paths = parameters.get('traversal_paths', self._default_traversal_paths)
            batch_size = parameters.get('batch_size', self._default_batch_size)
            # traverse thought all documents which have to be processed
            flat_docs = docs.traverse_flat(traversal_paths)
            # filter out documents without images
            filtered_docs = [doc for doc in flat_docs if doc.blob is not None]
            return _batch_generator(filtered_docs, batch_size)

        if not docs:
            return


        # traverse through a generator of batches of docs
        for docs_batch in _get_input_data(docs, parameters):
            # the blob dimension of imgs/cars.jpg at this point is (2, 681, 1264, 3)
            # Ensure the color channel axis is the default axis. i.e. c comes first
            # e.g. (h,w,c) -> (c,h,w) / (b,h,w,c) -> (b,c,h,w)
            blob_batch = [_move_channel_axis(d.blob, -1,
                                       self._default_channel_axis) for d in docs_batch]
            all_predictions = self._predict(blob_batch)

            for doc, blob, predictions in zip(docs_batch, blob_batch, all_predictions):
                bboxes = predictions['boxes'].detach()
                scores = predictions['scores'].detach()
                labels = predictions['labels']
                if self.on_gpu:
                    bboxes = bboxes.cpu()
                    scores = scores.cpu()
                    labels = labels.cpu()
                img = _load_image(blob * 255, self._default_channel_axis)

                for bbox, score, label in zip(bboxes.numpy(), scores.numpy(), labels.numpy()):
                    if score >= self.confidence_threshold:
                        x0, y0, x1, y1 = bbox
                        # note that tensors are [H, W] while PIL Images are [W, H]
                        top, left = int(y0), int(x0)
                        # target size must be (h, w)
                        target_size = (int(y1) - int(y0), int(x1) - int(x0))
                        # at this point, raw_img has the channel axis at the default tensor one
                        _img, top, left = _crop_image(img, target_size=target_size, top=top, left=left, how='precise')
                        _img = np.asarray(_img).astype('float32')
                        label_name = self.label_name_map[label]
                        self.logger.debug(
                            f'detected {label_name} with confidence {score} at position {(top, left)} and size {target_size}')

                        # a chunk is created for each of the objects detected for each image
                        d = Document(offset=0, weight=1., blob = _img, location=[top, left], tags={'label': label_name})
                        doc.chunks.append(d)
