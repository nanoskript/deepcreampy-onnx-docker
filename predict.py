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

# This is not configurable.
BATCH_SIZE = 8


def run_predictions(requests: list, is_mosaic=bool):
    results = []
    for start_index in range(0, len(requests), BATCH_SIZE):
        batch = requests[start_index:start_index + BATCH_SIZE]
        batch_size = len(batch)

        # Fill to required size.
        while len(batch) < BATCH_SIZE:
            batch.append(batch[0])

        censored, mask = list(zip(*batch))
        censored = np.float32(censored)
        mask = np.float32(mask)

        session = session_mosaic if is_mosaic else session_bar
        results += list(session.run(["add:0"], {
            "Placeholder:0": censored,
            "Placeholder_1:0": censored,
            "Placeholder_2:0": mask,
        })[0])[:batch_size]
    return results
