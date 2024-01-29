# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class GridSample(Base):
    @staticmethod
    def export_gridsample() -> None:
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="linear",
            padding_mode="zeros",
            align_corners=0,
        )
        # X shape, [N, C, H, W] - [1, 1, 4, 4]
        X = np.array(
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0, 15.0],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 6, 6, 2]
        Grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.6000, -1.0000],
                        [-0.2000, -1.0000],
                        [0.2000, -1.0000],
                        [0.6000, -1.0000],
                        [1.0000, -1.0000],
                    ],
                    [
                        [-1.0000, -0.6000],
                        [-0.6000, -0.6000],
                        [-0.2000, -0.6000],
                        [0.2000, -0.6000],
                        [0.6000, -0.6000],
                        [1.0000, -0.6000],
                    ],
                    [
                        [-1.0000, -0.2000],
                        [-0.6000, -0.2000],
                        [-0.2000, -0.2000],
                        [0.2000, -0.2000],
                        [0.6000, -0.2000],
                        [1.0000, -0.2000],
                    ],
                    [
                        [-1.0000, 0.2000],
                        [-0.6000, 0.2000],
                        [-0.2000, 0.2000],
                        [0.2000, 0.2000],
                        [0.6000, 0.2000],
                        [1.0000, 0.2000],
                    ],
                    [
                        [-1.0000, 0.6000],
                        [-0.6000, 0.6000],
                        [-0.2000, 0.6000],
                        [0.2000, 0.6000],
                        [0.6000, 0.6000],
                        [1.0000, 0.6000],
                    ],
                    [
                        [-1.0000, 1.0000],
                        [-0.6000, 1.0000],
                        [-0.2000, 1.0000],
                        [0.2000, 1.0000],
                        [0.6000, 1.0000],
                        [1.0000, 1.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 6, 6]
        Y = np.array(
            [
                [
                    [
                        [0.0000, 0.1500, 0.5500, 0.9500, 1.3500, 0.7500],
                        [0.6000, 1.5000, 2.3000, 3.1000, 3.9000, 2.1000],
                        [2.2000, 4.7000, 5.5000, 6.3000, 7.1000, 3.7000],
                        [3.8000, 7.9000, 8.7000, 9.5000, 10.3000, 5.3000],
                        [5.4000, 11.1000, 11.9000, 12.7000, 13.5000, 6.9000],
                        [3.0000, 6.1500, 6.5500, 6.9500, 7.3500, 3.7500],
                    ]
                ]
            ],
            dtype=np.float32,
        )
        expect(node, inputs=[X, Grid], outputs=[Y], name="test_gridsample")

    @staticmethod
    def export_gridsample_paddingmode() -> None:
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-10.0000, -10.0000],
                        [-5.0000, -5.0000],
                        [-0.2000, -0.2000],
                        [10.0000, 10.0000],
                    ],
                    [
                        [10.0000, 10.0000],
                        [-0.2000, -0.2000],
                        [5.0000, 5.0000],
                        [10.0000, 10.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        # setting padding_mode = 'zeros'
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            padding_mode="zeros",
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_zeros = np.array(
            [[[[0.0000, 0.0000, 1.7000, 0.0000], [0.0000, 1.7000, 0.0000, 0.0000]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_zeros],
            name="test_gridsample_zeros_padding",
        )

        # setting padding_mode = 'border'
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            padding_mode="border",
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_border = np.array(
            [[[[0.0000, 0.0000, 1.7000, 5.0000], [5.0000, 1.7000, 5.0000, 5.0000]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_border],
            name="test_gridsample_border_padding",
        )

        # setting padding_mode = 'reflection'
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            padding_mode="reflection",
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_reflection = np.array(
            [[[[2.5000, 0.0000, 1.7000, 2.5000], [2.5000, 1.7000, 5.0000, 2.5000]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_reflection],
            name="test_gridsample_reflection_padding",
        )

    @staticmethod
    def export_gridsample_mode_aligncorners() -> None:
        # X shape, [N, C, H, W] - [1, 1, 3, 2]
        X = np.array(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]],
            dtype=np.float32,
        )
        # Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
        Grid = np.array(
            [
                [
                    [
                        [-1.0000, -1.0000],
                        [-0.5000, -0.5000],
                        [-0.2000, -0.2000],
                        [0.0000, 0.0000],
                    ],
                    [
                        [0.0000, 0.0000],
                        [-0.2000, -0.2000],
                        [0.5000, 0.5000],
                        [1.0000, 1.0000],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        # setting mode = 'bilinear', default align_corners = 0
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="linear",
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bilinear = np.array(
            [[[[0.0000, 0.5000, 1.7000, 2.5000], [2.5000, 1.7000, 4.5000, 1.2500]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_bilinear],
            name="test_gridsample_bilinear",
        )

        # setting mode = 'bilinear', align_corners = 1
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="linear",
            align_corners=1,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_align_corners = np.array(
            [[[[0.0000, 1.2500, 2.0000, 2.5000], [2.5000, 2.0000, 3.7500, 5.0000]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_align_corners],
            name="test_gridsample_aligncorners_true",
        )

        # setting mode = 'nearest'
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="nearest",
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_nearest = np.array(
            [[[[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 5.0, 0.0]]]],
            dtype=np.float32,
        )

        expect(
            node, inputs=[X, Grid], outputs=[Y_nearest], name="test_gridsample_nearest"
        )

        # setting mode = 'bicubic'
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="cubic",
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bicubic = np.array(
            [[[[-0.1406, 0.3828, 1.7556, 2.9688], [2.9688, 1.7556, 5.1445, 1.3906]]]],
            dtype=np.float32,
        )

        expect(
            node, inputs=[X, Grid], outputs=[Y_bicubic], name="test_gridsample_bicubic"
        )

        # ============================================================================
        # Additional tests
        # The reference output tensors were generated using PyTorch 2.0.
        Grid = np.array(
            [
                [
                    [[-1.0, -0.8], [-0.6, -0.5], [-0.1, -0.2], [0.7, 0.0]],
                    [[0.0, 0.4], [0.2, -0.2], [-0.3, 0.5], [-1.0, 1.0]],
                ]
            ],
            dtype=np.float32,
        )

        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="nearest",
            align_corners=0,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_nearest = np.array(
            [[[[0.0, 0.0, 2.0, 3.0], [4.0, 3.0, 4.0, 4.0]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_nearest],
            name="test_gridsample_nearest_align_corners_0_additional_1",
        )

        # setting mode = 'nearest'
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="nearest",
            align_corners=1,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_nearest = np.array(
            [[[[0.0, 0.0, 2.0, 3.0], [2.0, 3.0, 4.0, 4.0]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_nearest],
            name="test_gridsample_nearest_align_corners_1_additional_1",
        )

        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="linear",
            align_corners=0,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bilinear = np.array(
            [[[[0.0000, 0.4500, 1.8000, 2.4000], [3.7000, 2.1000, 3.7000, 1.0000]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_bilinear],
            name="test_gridsample_bilinear_align_corners_0_additional_1",
        )

        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="linear",
            align_corners=1,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bilinear = np.array(
            [[[[0.4000, 1.2000, 2.0500, 2.8500], [3.3000, 2.2000, 3.3500, 4.0000]]]],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_bilinear],
            name="test_gridsample_bilinear_align_corners_1_additional_1",
        )

        # These two new bicubic tests produces slightly higher error ~5e-5
        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="cubic",
            align_corners=0,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bicubic = np.array(
            [
                [
                    [
                        [-0.173250, 0.284265, 1.923106, 2.568000],
                        [5.170375, 2.284414, 4.744844, 1.046875],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_bicubic],
            name="test_gridsample_bicubic_align_corners_0_additional_1",
        )

        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="cubic",
            align_corners=1,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bicubic = np.array(
            [
                [
                    [
                        [0.304001, 1.128750, 2.266270, 3.144844],
                        [4.531500, 2.455360, 4.599819, 4.000000],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_bicubic],
            name="test_gridsample_bicubic_align_corners_1_additional_1",
        )

    @staticmethod
    def export_volumeetric_gridsample_mode_aligncorners() -> None:
        X = np.array(
            [
                [
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                        [[9.0, 10.0], [11.0, 12.0]],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        Grid = np.array(
            [
                [
                    [
                        [[-1.0, -1.0, -1.0], [-1.0, -0.5, 0.3]],
                        [[-0.5, -0.5, -0.5], [1.0, -0.6, -1.0]],
                        [[-0.2, -0.2, -0.2], [0.4, 0.2, 0.6]],
                        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                    ],
                    [
                        [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]],
                        [[-0.2, -0.2, -0.2], [1.0, 0.4, -0.2]],
                        [[0.5, 0.5, 0.5], [-1.0, -0.8, 0.8]],
                        [[1.0, 1.0, 1.0], [0.4, 0.6, -0.3]],
                    ],
                ]
            ],
            dtype=np.float32,
        )

        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="nearest",
            align_corners=0,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_nearest = np.array(
            [
                [
                    [
                        [[1.0, 5.0], [1.0, 0.0], [5.0, 12.0], [5.0, 5.0]],
                        [[5.0, 0.0], [5.0, 0.0], [12.0, 9.0], [0.0, 8.0]],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_nearest],
            name="test_gridsample_volumetric_nearest_align_corners_0",
        )

        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="nearest",
            align_corners=1,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_nearest = np.array(
            [
                [
                    [
                        [[1.0, 5.0], [1.0, 2.0], [5.0, 12.0], [5.0, 5.0]],
                        [[5.0, 7.0], [5.0, 8.0], [12.0, 9.0], [12.0, 8.0]],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_nearest],
            name="test_gridsample_volumetric_nearest_align_corners_1",
        )

        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="linear",
            align_corners=0,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bilinear = np.array(
            [
                [
                    [
                        [
                            [0.1250, 3.4000],
                            [2.0000, 0.4500],
                            [4.7000, 10.9000],
                            [6.5000, 3.0000],
                        ],
                        [
                            [6.5000, 1.7500],
                            [4.7000, 3.3000],
                            [11.0000, 2.5200],
                            [1.5000, 5.4900],
                        ],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_bilinear],
            name="test_gridsample_volumetric_bilinear_align_corners_0",
        )

        node = onnx.helper.make_node(
            "GridSample",
            inputs=["X", "Grid"],
            outputs=["Y"],
            mode="linear",
            align_corners=1,
        )
        # Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
        Y_bilinear = np.array(
            [
                [
                    [
                        [
                            [1.0000, 6.7000],
                            [3.7500, 2.4000],
                            [5.4000, 9.3000],
                            [6.5000, 6.0000],
                        ],
                        [
                            [6.5000, 7.0000],
                            [5.4000, 6.6000],
                            [9.2500, 8.4000],
                            [12.0000, 6.1000],
                        ],
                    ]
                ]
            ],
            dtype=np.float32,
        )

        expect(
            node,
            inputs=[X, Grid],
            outputs=[Y_bilinear],
            name="test_gridsample_volumetric_bilinear_align_corners_1",
        )

    """
    For someone who want to test by script. Comment it cause github ONNX CI
    do not have the torch python package.
    @staticmethod
    def export_gridsample_torch():  # type: () -> None
        node = onnx.helper.make_node(
            'GridSample',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=0,
        )

        # X shape, [N, C, H, W] - [1, 1, 4, 4]
        # Grid shape, [N, H_out, W_out, 2] - [1, 6, 6, 2]
        # Y shape, [N, C, H_out, W_out] - [1, 1, 6, 6]
        import torch
        X = torch.arange(3 * 3).view(1, 1, 3, 3).float()
        d = torch.linspace(-1, 1, 6)
        meshx, meshy = torch.meshgrid((d, d))
        grid = torch.stack((meshy, meshx), 2)
        Grid = grid.unsqueeze(0)
        Y = torch.nn.functional.grid_sample(X, Grid, mode='bilinear',
                                            padding_mode='zeros', align_corners=False)
        expect(node, inputs=[X.numpy(), Grid.numpy()], outputs=[Y.numpy()],
               name='test_gridsample_torch')
    """
