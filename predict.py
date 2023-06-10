import numpy as np
import tensorflow as tf
import onnxruntime
from onnxruntime_extensions import onnx_op, PyCustomOpDef, get_library_path

tf = tf.compat.v1
tf.enable_eager_execution()


@onnx_op(op_type="ExtractImagePatches",
         inputs=[PyCustomOpDef.dt_float,
                 PyCustomOpDef.dt_int32,
                 PyCustomOpDef.dt_int32,
                 PyCustomOpDef.dt_int32,
                 PyCustomOpDef.dt_string],
         outputs=[PyCustomOpDef.dt_float])
def extract_image_patches(arr, ksizes, strides, rates, padding):
    return tf.extract_image_patches(
        arr,
        ksizes.tolist(),
        strides.tolist(),
        rates.tolist(),
        padding[0]
    ).numpy()


session_options = onnxruntime.SessionOptions()
session_options.register_custom_ops_library(get_library_path())
session_bar = onnxruntime.InferenceSession("./vendor/bar.onnx", session_options)
session_mosaic = onnxruntime.InferenceSession("./vendor/mosaic.onnx", session_options)


def predict(censored, mask, is_mosaic=bool):
    censored = np.float32([censored])
    mask = np.float32([mask])

    session = session_mosaic if is_mosaic else session_bar
    return list(session.run(["add:0"], {
        "Placeholder:0": censored,
        "Placeholder_1:0": censored,
        "Placeholder_2:0": mask,
    })[0])[0]
