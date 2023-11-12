
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################
"""
Automatically converts a model to mixed precision, excluding the minimum number of nodes required to
ensure customized_validate_func returns True and/or results are equal according to rtol/atol and saves
the converted model on the disk. This function requires the source model's path as an input (source_model_path),
so it still works well when the model's size > 2G.
"""

import copy
import numpy as np
import onnx
import os
import uuid
from onnxconverter_common import float16
from onnx import shape_inference
from .auto_mixed_precision import SegmentList


def auto_convert_mixed_precision_model_path(source_model_path, input_feed,
                                            target_model_path, provider, location=None,
                                            customized_validate_func=None, rtol=None, atol=None,
                                            keep_io_types=True, verbose=False):
    """
    This tool converts a model to mixed precision (float32->float16) while excluding nodes as needed to maintain
    a certain accuracy. After the conversion, the model will be saved on the disk under given path.
    A model with a size > 2G should leverage this.

    Example usage:

        from onnxconverter_common import auto_mixed_precision_model_path
        import numpy as np
        import time

        source_model_path = "/home/user/onnx/mymodel/model/8_fp32/graph.onnx"
        target_model_path = '/home/user/onnx/mymodel/output/fp16.onnx'  # better to specify an %output% folder
        location = "fp16_tensor.data"

        # Could also use rtol/atol attributes directly instead of this
        def validate(res1, res2):
            for r1, r2 in zip(res1, res2):
                if not np.allclose(r1, r2, rtol=0.01, atol=0.001):
                    return False
            return True

    Parameters:

    - source_mode_path: the full or relative path of fp32 model.
    - input_feed: this function will use this input_feed to do inference/validation during the convertion.
    - target_model_path: the full or relative path where the fp16 model will be saved. make sure the volume is enough.
    - provider: should be ['CPUExecutionProvider'] when you want to inference on CPU machine finally,
                or ['CUDAExecutionProvider'] when you want to inference on CUDA machine finally.
    - location: the external data will be saved as %target_model_path/%location.
    - customized_validate_func: define customized validate function, must return True or False.
                                if customized_validate_func is None, will use np.allcose(r1,r2,rtol=1e-3,atol=1e-5).
    - rtol/atol: the relative or absolute tolerance to do validation.
    - keep_io_types: set to True, so the input_feed can be used for both fp32 and fp16 models during the conversion.
    - verbose: set to True to show more information during the convertion.

    You don't need to call onnx.save_model() after this function call in your code.
    """

    print("Step 0: checking input parameters...")

    if not isinstance(source_model_path, str):
        raise TypeError('auto_convert_mixed_precision_model_path only accepts model Path (String),'
                        'you can use auto_convert_mixed_precision for the ModelProto.')

    if not isinstance(input_feed, dict):
        raise ValueError("input_feed should be a dictionary such as {'modelInput': input_x.astype(np.float32)}")

    if rtol is None:
        rtol = 1e-3

    if atol is None:
        atol = 1e-5

    if location is None:
        print("Setting location as 'fp16_tensor.data'.")
        location = "fp16_tensor.data"

    if not os.path.exists(source_model_path):
        raise ValueError("source_model_path does not exist: %s" % source_model_path)

    try:
        print("Step 1: copy source model to working folder, then do basic checking...")

        tmp_model32_path, tmp_model32_tensor_name = generate_temp_filename(target_model_path)
        kwargs = {
            "tmp_model32_path": tmp_model32_path,
            "tmp_model32_tensor_name": tmp_model32_tensor_name,
            "source_model_path": source_model_path,
            "input_feed": input_feed,
            "target_model_path": target_model_path,
            "location": location,
            "customized_validate_func": customized_validate_func,
            "rtol": rtol,
            "atol": atol,
            "keep_io_types": keep_io_types,
            "providers": provider,
            "verbose": verbose
            }
        model_32, output_32 = _adjust_and_inference_source_model(**kwargs)

        print("Step 2: try to convert to fp16 model iteratively...")

        node_names = [n.name for n in model_32.graph.node if n.op_type not in ["Loop", "If", "Scan"]]
        kwargs["model_32"] = model_32
        kwargs["res1"] = output_32
        kwargs["node_block_list"] = node_names
        kwargs["is_final_model"] = False
        result = _convert_and_check_inference_result(**kwargs)
        if not result:
            raise ValueError("Validation failed for model with nothing converted to fp16. "
                             "Given parameters %r." % kwargs)

        final_block_list = _find_nodes_blocking_fp16(**kwargs)

        print("Step 3: Final converting...")
        kwargs["node_block_list"] = final_block_list
        kwargs["is_final_model"] = True
        valid = _convert_and_check_inference_result(**kwargs)
        if not valid:
            raise ValueError("Validation failed for final fp16 model.")

        print("Complete!")
        print("Your fp16 model is here %s and the external data file is here %s" % (target_model_path, location))

    finally:
        _clean_output_folder(tmp_model32_path, tmp_model32_tensor_name)


def generate_temp_filename(target_model_path):
    target_model_folder = os.path.dirname(target_model_path)
    if not os.path.exists(target_model_folder):
        os.mkdir(target_model_folder)
    tensor_filename = str(uuid.uuid1())
    onnx_filename = os.path.join(target_model_folder, tensor_filename + ".onnx")
    return onnx_filename, tensor_filename + ".data"


def _validate_result(**kwargs):
    customized_validate_func = kwargs.get("customized_validate_func")
    rtol = kwargs.get("rtol")
    atol = kwargs.get("atol")
    res1 = kwargs.get("res1")
    res2 = kwargs.get("res2")

    if customized_validate_func is not None:
        return customized_validate_func(res1, res2)
    else:
        for r1, r2 in zip(res1, res2):
            if not np.allclose(r1, r2, rtol=rtol, atol=atol):
                return False
        return True


def _adjust_and_inference_source_model(**kwargs):
    source_model_path = kwargs.get('source_model_path')
    input_feed = kwargs.get('input_feed')
    providers = kwargs.get('providers')
    tmp_model32_path = kwargs.get("tmp_model32_path")
    tmp_model32_tensor_name = kwargs.get("tmp_model32_tensor_name")

    model_32 = onnx.load(source_model_path)
    save_model(model_32, tmp_model32_path, location=tmp_model32_tensor_name)

    shape_inference.infer_shapes_path(tmp_model32_path)
    model_32 = onnx.load(tmp_model32_path)

    output_32 = inference(tmp_model32_path, input_feed, providers=providers)

    kwargs["res1"] = output_32
    kwargs["res2"] = output_32
    if not _validate_result(**kwargs):
        raise ValueError("validation failed for fp32 model")

    return model_32, output_32


def _find_nodes_blocking_fp16(**kwargs):
    node_names = kwargs.get('node_block_list')
    verbose = kwargs.get("verbose")

    segments = SegmentList(node_names)
    i = 0
    while segments.get_largest() is not None:
        seg = segments.get_largest()
        nodes_to_try = segments.get_nodes(seg)
        i += 1
        print("Running attempt %d excluding conversion of %s nodes" % (i, len(nodes_to_try)))
        kwargs["node_block_list"] = nodes_to_try
        if _convert_and_check_inference_result(**kwargs):
            seg.good = True
            if verbose:
                print("Attempt succeeded.")
        else:
            if verbose:
                print("Attempt failed.")
            if seg.size == 1:
                seg.bad = True
            else:
                seg.split()
        if verbose:
            print("segments=", segments)
    if verbose:
        print("Done! these nodes will keep float32 type:", segments.get_nodes())

    return segments.get_nodes()


def _convert_and_check_inference_result(**kwargs):
    model_32 = kwargs.get("model_32")
    keep_io_types = kwargs.get("keep_io_types")
    is_final_model = kwargs.get("is_final_model")
    target_model_path = kwargs.get("target_model_path")
    node_block_list = kwargs.get("node_block_list")
    input_feed = kwargs.get("input_feed")
    providers = kwargs.get("providers")
    tmp_model32_tensor_name = kwargs.get("tmp_model32_tensor_name")
    verbose = kwargs.get("verbose")

    if verbose:
        print("convert to float 16...")
        _print_node_block_list(node_block_list)
    model_16 = float16.convert_float_to_float16(
        copy.deepcopy(model_32), node_block_list=node_block_list,
        keep_io_types=keep_io_types, disable_shape_infer=True)

    if is_final_model:
        location = kwargs.get("location")  # using the speficified external data file name
    else:
        location = tmp_model32_tensor_name  # using temporary file name
    save_model(model_16, target_model_path, location=location)

    output_16 = inference(target_model_path, input_feed, providers=providers)
    kwargs["res2"] = output_16
    result = _validate_result(**kwargs)
    if verbose:
        print("validate result = ", result)
    return result


def inference(model_path, input_feed, providers=None):
    # delayed import to avoid taking a strong dependancy on onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, None, providers=providers)
    output = sess.run(None, input_feed)
    return output


def save_model(model, model_path, location=None):
    # remove the old one, because the save_model function will use append mode to
    # make the tensor data file too big if save many times
    _clean_output_folder(model_path, location)
    onnx.save_model(model, model_path, save_as_external_data=True, location=location)


def _print_node_block_list(node_block_list, max_len=128):
    print("node block list =")
    if (len(node_block_list) < max_len):
        print(node_block_list)
    else:
        tmp_list = node_block_list[0:64] + ['......'] + node_block_list[-64:]
        print(tmp_list)


def _clean_output_folder(tmp_model32_path, tmp_model32_tensor_name):
    if os.path.exists(tmp_model32_path):
        os.remove(tmp_model32_path)
    tmp_tensor_path = os.path.join(os.path.dirname(tmp_model32_path), tmp_model32_tensor_name)
    if os.path.exists(tmp_tensor_path):
        os.remove(tmp_tensor_path)
