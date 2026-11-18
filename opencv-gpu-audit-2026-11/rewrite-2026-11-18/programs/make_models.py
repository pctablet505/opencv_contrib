# Writes the ONNX models the measurement programs load. Run: python make_models.py <out_dir>
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

out = sys.argv[1]
rng = np.random.default_rng(0)
f32 = TensorProto.FLOAT


def save(name, nodes, inputs, outputs, inits, ms=False):
    opsets = [helper.make_opsetid("", 17)]
    if ms:
        opsets.append(helper.make_opsetid("com.microsoft", 1))
    model = helper.make_model(helper.make_graph(nodes, name, inputs, outputs, initializer=inits),
                              opset_imports=opsets)
    model.ir_version = 8
    if not ms:
        onnx.checker.check_model(model)
    onnx.save(model, f"{out}/{name}.onnx")


def vi(name, dims, t=f32):
    return helper.make_tensor_value_info(name, t, dims)


# Gemm with a constant weight, for Net::setParam() on the ONNX graph path
save("gemm_setparam", [helper.make_node("Gemm", ["X", "W"], ["Y"])],
     [vi("X", [1, 4])], [vi("Y", [1, 4])],
     [numpy_helper.from_array(np.eye(4, dtype=np.float32) * 2, "W")])

# MatMul + Add with a [1, 1, N] bias on a 3-D input
w = rng.uniform(-1, 1, (8, 6)).astype(np.float32)
save("matmul_bias_1x1xN", [helper.make_node("MatMul", ["X", "W"], ["MM"]), helper.make_node("Add", ["MM", "B"], ["Y"])],
     [vi("X", [2, 4, 8])], [vi("Y", [2, 4, 6])],
     [numpy_helper.from_array(w, "W"), numpy_helper.from_array(np.arange(1, 7, dtype=np.float32).reshape(1, 1, 6), "B")])

# MatMul + Add with an [M, 1] column bias, M == N == 3
w3 = rng.uniform(-1, 1, (4, 3)).astype(np.float32)
save("matmul_bias_Mx1", [helper.make_node("MatMul", ["X", "W"], ["MM"]), helper.make_node("Add", ["MM", "B"], ["Y"])],
     [vi("X", [3, 4])], [vi("Y", [3, 3])],
     [numpy_helper.from_array(w3, "W"), numpy_helper.from_array(np.array([[1], [2], [3]], dtype=np.float32), "B")])

# com.microsoft Attention, hidden 768, 12 heads, one model per sequence length (OpenCV requires qkv_hidden_sizes)
D, H = 768, 12
wa = (rng.standard_normal((D, 3 * D)) * 0.02).astype(np.float32)
ba = (rng.standard_normal(3 * D) * 0.02).astype(np.float32)
for S in (256, 512, 1024, 2048, 4096):
    save(f"attention_S{S}", [helper.make_node("Attention", ["X", "W", "Bias"], ["Y"], domain="com.microsoft", num_heads=H, qkv_hidden_sizes=[D, D, D])],
         [vi("X", [1, S, D])], [vi("Y", [1, S, D])],
         [numpy_helper.from_array(wa, "W"), numpy_helper.from_array(ba, "Bias")], ms=True)

# com.microsoft Attention with the spec's int32 mask_index [B, S] (1 = attend, 0 = padding)
S, Dm, Hm = 16, 64, 4
wm = (rng.standard_normal((Dm, 3 * Dm)) * 0.1).astype(np.float32)
bm = (rng.standard_normal(3 * Dm) * 0.1).astype(np.float32)
mask = np.ones((1, S), dtype=np.int32)
mask[0, 12:] = 0
attn = helper.make_node("Attention", ["X", "W", "Bias", "M"], ["Y"], domain="com.microsoft", num_heads=Hm, qkv_hidden_sizes=[Dm, Dm, Dm])
save("attention_mask_input", [attn], [vi("X", [1, S, Dm]), vi("M", [1, S], TensorProto.INT32)], [vi("Y", [1, S, Dm])],
     [numpy_helper.from_array(wm, "W"), numpy_helper.from_array(bm, "Bias")], ms=True)
save("attention_mask_const", [attn], [vi("X", [1, S, Dm])], [vi("Y", [1, S, Dm])],
     [numpy_helper.from_array(wm, "W"), numpy_helper.from_array(bm, "Bias"), numpy_helper.from_array(mask, "M")], ms=True)

# com.microsoft Attention without a bias, which the spec allows: two inputs, and an empty bias name
for name, ins in (("attention_nobias", ["X", "W"]), ("attention_emptybias", ["X", "W", ""])):
    save(name, [helper.make_node("Attention", ins, ["Y"], domain="com.microsoft", num_heads=Hm, qkv_hidden_sizes=[Dm, Dm, Dm])],
         [vi("X", [1, S, Dm])], [vi("Y", [1, S, Dm])], [numpy_helper.from_array(wm, "W")], ms=True)

# onnxruntime references for the mask models, written as raw float32
import onnxruntime as ort
x = rng.uniform(-1, 1, (1, S, Dm)).astype(np.float32)
x.tofile(f"{out}/attention_mask_x.bin")
mask.tofile(f"{out}/attention_mask_m.bin")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
ref = ort.InferenceSession(f"{out}/attention_mask_input.onnx", so, providers=["CPUExecutionProvider"]).run(None, {"X": x, "M": mask})[0]
ref.astype(np.float32).tofile(f"{out}/attention_mask_ref.bin")
nomask = ort.InferenceSession(f"{out}/attention_mask_const.onnx", so, providers=["CPUExecutionProvider"]).run(None, {"X": x})[0]
print("ort mask_input vs mask_const max diff", float(np.abs(ref - nomask).max()))
print("written", out)

# 24 LayerNormalization nodes in a row on [1, 1024, 768], for timing the CUDA kernel
Dl, Sl = 768, 1024
nodes, inits = [], []
prev = "X"
for i in range(24):
    inits += [numpy_helper.from_array(np.ones(Dl, np.float32), f"g{i}"), numpy_helper.from_array(np.zeros(Dl, np.float32), f"b{i}")]
    out_name = "Y" if i == 23 else f"t{i}"
    nodes.append(helper.make_node("LayerNormalization", [prev, f"g{i}", f"b{i}"], [out_name], axis=-1, epsilon=1e-5))
    prev = out_name
save("layernorm_x24", nodes, [vi("X", [1, Sl, Dl])], [vi("Y", [1, Sl, Dl])], inits)
print("written layernorm_x24")
