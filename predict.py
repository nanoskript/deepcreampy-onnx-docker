import numpy as np
import onnxruntime

session_options = onnxruntime.SessionOptions()
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
