# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import dataclasses
from typing import Optional, Tuple

import numpy as np

from onnx.reference.op_run import OpRun


@dataclasses.dataclass
class PrepareContext:
    boxes_data_: Optional[np.ndarray] = None
    boxes_size_: int = 0
    scores_data_: Optional[np.ndarray] = None
    scores_size_: int = 0
    max_output_boxes_per_class_: Optional[np.ndarray] = None
    score_threshold_: Optional[np.ndarray] = None
    iou_threshold_: Optional[np.ndarray] = None
    num_batches_: int = 0
    num_classes_: int = 0
    num_boxes_: int = 0


class SelectedIndex:
    __slots__ = ("batch_index_", "class_index_", "box_index_")

    def __init__(
        self, batch_index: int = 0, class_index: int = 0, box_index: int = 0
    ) -> None:
        self.batch_index_ = batch_index
        self.class_index_ = class_index
        self.box_index_ = box_index


def max_min(lhs: float, rhs: float) -> Tuple[float, float]:
    if lhs >= rhs:
        return rhs, lhs
    return lhs, rhs


def suppress_by_iou(  # noqa: PLR0911
    boxes_data: np.ndarray,
    box_index1: int,
    box_index2: int,
    center_point_box: int,
    iou_threshold: float,
) -> bool:
    box1 = boxes_data[box_index1]
    box2 = boxes_data[box_index2]
    # center_point_box_ only support 0 or 1
    if center_point_box == 0:
        # boxes data format [y1, x1, y2, x2]
        x1_min, x1_max = max_min(box1[1], box1[3])
        x2_min, x2_max = max_min(box2[1], box2[3])

        intersection_x_min = max(x1_min, x2_min)
        intersection_x_max = min(x1_max, x2_max)
        if intersection_x_max <= intersection_x_min:
            return False

        y1_min, y1_max = max_min(box1[0], box1[2])
        y2_min, y2_max = max_min(box2[0], box2[2])
        intersection_y_min = max(y1_min, y2_min)
        intersection_y_max = min(y1_max, y2_max)
        if intersection_y_max <= intersection_y_min:
            return False
    else:
        # 1 == center_point_box_ => boxes data format [x_center, y_center, width, height]
        box1_width_half = box1[2] / 2
        box1_height_half = box1[3] / 2
        box2_width_half = box2[2] / 2
        box2_height_half = box2[3] / 2

        x1_min = box1[0] - box1_width_half
        x1_max = box1[0] + box1_width_half
        x2_min = box2[0] - box2_width_half
        x2_max = box2[0] + box2_width_half

        intersection_x_min = max(x1_min, x2_min)
        intersection_x_max = min(x1_max, x2_max)
        if intersection_x_max <= intersection_x_min:
            return False

        y1_min = box1[1] - box1_height_half
        y1_max = box1[1] + box1_height_half
        y2_min = box2[1] - box2_height_half
        y2_max = box2[1] + box2_height_half

        intersection_y_min = max(y1_min, y2_min)
        intersection_y_max = min(y1_max, y2_max)
        if intersection_y_max <= intersection_y_min:
            return False

    intersection_area = (intersection_x_max - intersection_x_min) * (
        intersection_y_max - intersection_y_min
    )
    if intersection_area <= 0:
        return False

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersection_area

    if area1 <= 0 or area2 <= 0 or union_area <= 0:
        return False

    intersection_over_union = intersection_area / union_area
    return intersection_over_union > iou_threshold


class BoxInfo:
    def __init__(self, score: float = 0, idx: int = -1):
        self.score_ = score
        self.idx_ = idx

    def __lt__(self, rhs) -> bool:  # type: ignore
        return self.score_ < rhs.score_ or (  # type: ignore
            self.score_ == rhs.score_ and self.idx_ > rhs.idx_
        )

    def __repr__(self) -> str:
        return f"BoxInfo({self.score_}, {self.idx_})"


class NonMaxSuppression(OpRun):
    def get_thresholds_from_inputs(
        self,
        pc: PrepareContext,
        max_output_boxes_per_class: int,
        iou_threshold: float,
        score_threshold: float,
    ) -> Tuple[int, float, float]:
        if pc.max_output_boxes_per_class_ is not None:
            max_output_boxes_per_class = max(pc.max_output_boxes_per_class_[0], 0)

        if pc.iou_threshold_ is not None:
            iou_threshold = pc.iou_threshold_[0]

        if pc.score_threshold_ is not None:
            score_threshold = pc.score_threshold_[0]

        return max_output_boxes_per_class, iou_threshold, score_threshold

    def prepare_compute(  # type: ignore
        self,
        pc: PrepareContext,
        boxes_tensor: np.ndarray,  # float
        scores_tensor: np.ndarray,  # float
        max_output_boxes_per_class_tensor: np.ndarray,  # int
        iou_threshold_tensor: np.ndarray,  # float
        score_threshold_tensor: np.ndarray,  # float
    ):
        pc.boxes_data_ = boxes_tensor
        pc.scores_data_ = scores_tensor

        if max_output_boxes_per_class_tensor.size != 0:
            pc.max_output_boxes_per_class_ = max_output_boxes_per_class_tensor
        if iou_threshold_tensor.size != 0:
            pc.iou_threshold_ = iou_threshold_tensor
        if score_threshold_tensor.size != 0:
            pc.score_threshold_ = score_threshold_tensor

        pc.boxes_size_ = boxes_tensor.size
        pc.scores_size_ = scores_tensor.size

        boxes_dims = boxes_tensor.shape
        scores_dims = scores_tensor.shape

        pc.num_batches_ = boxes_dims[0]
        pc.num_classes_ = scores_dims[1]
        pc.num_boxes_ = boxes_dims[1]

    def _run(  # type: ignore
        self,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        center_point_box,
    ):
        center_point_box = center_point_box or self.center_point_box  # type: ignore

        pc = PrepareContext()
        self.prepare_compute(
            pc,
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )

        (
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        ) = self.get_thresholds_from_inputs(pc, 0, 0, 0)

        if max_output_boxes_per_class.size == 0:
            return (np.empty((0,), dtype=np.int64),)

        boxes_data = pc.boxes_data_
        scores_data = pc.scores_data_

        selected_indices = []
        # std::vector<BoxInfo> selected_boxes_inside_class;
        # selected_boxes_inside_class.reserve(
        #    std::min<size_t>(static_cast<size_t>(max_output_boxes_per_class), pc.num_boxes_));

        for batch_index in range(pc.num_batches_):
            for class_index in range(pc.num_classes_):
                box_score_offset = (batch_index, class_index)
                batch_boxes = boxes_data[batch_index]  # type: ignore
                # std::vector<BoxInfo> candidate_boxes;
                # candidate_boxes.reserve(pc.num_boxes_);

                # Filter by score_threshold_
                candidate_boxes = []
                class_scores = scores_data[box_score_offset]  # type: ignore
                if pc.score_threshold_ is not None:
                    for box_index in range(pc.num_boxes_):
                        if class_scores[box_index] > score_threshold:
                            candidate_boxes.append(
                                BoxInfo(class_scores[box_index], box_index)
                            )
                else:
                    for box_index in range(pc.num_boxes_):
                        candidate_boxes.append(
                            BoxInfo(class_scores[box_index], box_index)
                        )

                sorted_boxes = sorted(candidate_boxes)

                selected_boxes_inside_class = []  # type: ignore
                # Get the next box with top score, filter by iou_threshold.
                while (
                    len(sorted_boxes) > 0
                    and len(selected_boxes_inside_class) < max_output_boxes_per_class
                ):
                    next_top_score = sorted_boxes[-1]

                    selected = True
                    # Check with existing selected boxes for this class,
                    # suppress if exceed the IOU (Intersection Over Union) threshold.
                    for selected_index in selected_boxes_inside_class:
                        if suppress_by_iou(
                            batch_boxes,
                            next_top_score.idx_,
                            selected_index.idx_,
                            center_point_box,
                            iou_threshold,
                        ):
                            selected = False
                            break

                    if selected:
                        selected_boxes_inside_class.append(next_top_score)
                        selected_indices.append(
                            SelectedIndex(batch_index, class_index, next_top_score.idx_)
                        )

                    sorted_boxes.pop()

        result = np.empty((len(selected_indices), 3), dtype=np.int64)
        for i in range(result.shape[0]):
            result[i, 0] = selected_indices[i].batch_index_
            result[i, 1] = selected_indices[i].class_index_
            result[i, 2] = selected_indices[i].box_index_
        return (result,)
