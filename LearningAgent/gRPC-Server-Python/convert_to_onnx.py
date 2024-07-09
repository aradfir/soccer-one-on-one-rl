import torch as th
from typing import Tuple
import sys
import torch
import onnx
from stable_baselines3 import DDPG
import numpy as np
import onnxruntime as ort
import onnx
import onnxruntime as ort
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.policies import BasePolicy

def rescale_action(action:np.float32,low=-1,high=1):
    action_normalized = (action + 1) / 2  # Normalize to [0, 1]
    action_mapped = action_normalized * (high-low) + low  # Map to [0.15, 1]
    return action_mapped



if __name__ == "__main__":
    model_path = sys.argv[1]
    model = DDPG.load(model_path, device="cpu")
    observation_size = model.observation_space.shape
    observation = np.zeros((1, *observation_size)).astype(np.float32)
    for i in range(len(observation[0])):
        observation[0][i] = 0.5
    print("Original DDPG:", model.predict(observation, deterministic=True))
    dummy_input = th.randn(1, *observation_size)
    # for future reference: exporting model.policy may not be enough.
    # see: https://stable-baselines3.readthedocs.io/en/master/guide/export.html#export-to-onnx
    th.onnx.export(
        model.policy,
        dummy_input,
        "model.onnx",
        opset_version=17,
        input_names=["input"],
    )
    print("Model converted to ONNX format")
    # Load the ONNX model and verify input/output names


    onnx_path = "model.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_sess = ort.InferenceSession(onnx_path)
    out:list = ort_sess.run(None, {"input": observation})
    print(len(out[0][0]))
    out[0][0][1] = rescale_action(out[0][0][1],0.15,1.0)
    print(out)






