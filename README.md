# deepcreampy-onnx-docker

[Docker Hub](https://hub.docker.com/r/nanoskript/deepcreampy)
| [API documentation](https://deepcreampy.nanoskript.dev/docs)

ONNX model and Docker service for DeepCreamPy.

Exporting DeepCreamPy's models from Tensorflow to ONNX improves inference
times when running on CPU and significantly reduces memory usage. Large parts
of the model graph that are not useful for inference are eliminated through
the conversion process.

This repository uses [Deepshift's mirror](https://github.com/Deepshift/DeepCreamPy) of DeepCreamPy.

## Docker installation

```
docker run --publish $PORT:$PORT --env PORT=$PORT --detach nanoskript/deepcreampy
```

