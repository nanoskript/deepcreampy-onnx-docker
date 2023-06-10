import onnx
import tf2onnx
from tf2onnx import utils, constants
from tf2onnx.graph import GraphUtil
from onnx.numpy_helper import from_array
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


def prune_onnx_model(model):
    def find_node_by_name(nodes, name):
        for node in nodes:
            if node.name == name:
                return node

    def find_initializer_by_name(graph, name):
        for initializer in graph.initializer:
            if initializer.name == name:
                return initializer

    # Prune duplicated network for batching.
    nodes = model.graph.node
    block_end = find_node_by_name(nodes, "CB1/concat_6")
    del block_end.input[1:]

    # Reshape for batch size of 1.
    reshape = find_node_by_name(nodes, "CB1/Reshape_1")
    reshape_const = find_initializer_by_name(model.graph, reshape.input[1])
    reshape_const.CopyFrom(from_array(np.int64([1, 1, 1, 900, 2304]), reshape_const.name))

    # Change external dimensions.
    for node in list(model.graph.input) + list(model.graph.output):
        node.type.tensor_type.shape.dim[0].dim_value = 1

    # Dangling nodes must be eliminated.
    model = GraphUtil.optimize_model_proto(model)
    onnx.checker.check_model(model, full_check=True)
    return model


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
        model = prune_onnx_model(onnx_graph)
        f.write(model.SerializeToString())


def main():
    convert_model("./models/bar/Train_775000.meta", "./vendor/bar.onnx")
    convert_model("./models/mosaic/Train_290000.meta", "./vendor/mosaic.onnx")


if __name__ == '__main__':
    main()
