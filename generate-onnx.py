import tf2onnx
from tf2onnx import utils, constants
import numpy as np


def convert_extract_image_patches(ctx, node, _name, _args):
    node.type = "ExtractImagePatches"
    node.domain = constants.CONTRIB_OPS_DOMAIN

    ksizes = node.get_attr_value("ksizes")
    strides = node.get_attr_value("strides")
    rates = node.get_attr_value("rates")
    padding = node.get_attr_str("padding")

    ksizes_const = ctx.make_const(utils.make_name("ksizes_const"), np.array(ksizes, dtype=np.int32))
    strides_const = ctx.make_const(utils.make_name("strides_const"), np.array(strides, dtype=np.int32))
    rates_const = ctx.make_const(utils.make_name("rates_const"), np.array(rates, dtype=np.int32))
    padding_cost = ctx.make_const(utils.make_name("padding_const"), np.array([padding], dtype=object))

    for key in list(node.attr.keys()):
        del node.attr[key]

    ctx.replace_inputs(node, node.input + [
        ksizes_const.output[0],
        strides_const.output[0],
        rates_const.output[0],
        padding_cost.output[0],
    ])


def convert_model(checkpoint, target):
    inputs = ["Placeholder:0", "Placeholder_1:0", "Placeholder_2:0"]
    outputs = ["add:0"]

    graph, inputs, outputs = tf2onnx.tf_loader.from_checkpoint(checkpoint, inputs, outputs)
    (onnx_graph, _) = tf2onnx.convert.from_graph_def(
        graph, input_names=inputs, output_names=outputs,
        custom_op_handlers={"ExtractImagePatches": (convert_extract_image_patches, [])},
        extra_opset=[utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)],
    )

    with open(target, "wb") as f:
        f.write(onnx_graph.SerializeToString())


def main():
    convert_model("./models/bar/Train_775000.meta", "./vendor/bar.onnx")
    convert_model("./models/mosaic/Train_290000.meta", "./vendor/mosaic.onnx")


if __name__ == '__main__':
    main()
