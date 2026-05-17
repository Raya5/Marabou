import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn.functional as F


# --------- helpers: ONNX weights -> torch ---------

def _get_initializers(onnx_model):
    init = {}
    for t in onnx_model.graph.initializer:
        # init[t.name] = torch.from_numpy(numpy_helper.to_array(t)).float()
        arr = numpy_helper.to_array(t)
        init[t.name] = torch.from_numpy(arr.copy()).float()

    return init


def _as_torch(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.from_numpy(np.asarray(x)).float()


# --------- IBP primitives ---------

def _ibp_linear(l, u, W, b=None):
    # y = x @ W^T + b   (PyTorch Linear convention)
    c = (l + u) * 0.5
    r = (u - l) * 0.5
    W_abs = W.abs()
    yc = c.matmul(W.t())
    yr = r.matmul(W_abs.t())
    if b is not None:
        yc = yc + b
    return yc - yr, yc + yr


def _ibp_conv2d(l, u, W, b=None, stride=1, padding=0, dilation=1, groups=1):
    # Conv2d: (N,C,H,W)
    c = (l + u) * 0.5
    r = (u - l) * 0.5
    yc = F.conv2d(c, W, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
    yr = F.conv2d(r, W.abs(), None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    return yc - yr, yc + yr


def _ibp_relu(l, u):
    return F.relu(l), F.relu(u)


# --------- minimal ONNX IBP runner ---------

def ibp_onnx_logits_bounds(onnx_path, x_l, x_u, input_name=None, output_name=None):
    """
    Returns (logits_lower, logits_upper) for a batch of boxes.
    x_l, x_u: torch tensors with identical shape, including batch dim.
    Assumes model outputs logits (pre-softmax). (Paper also notes this for Marabou usage.) 
    """
    model = onnx.load(onnx_path)
    init = _get_initializers(model)

    # map from tensor name -> (l,u)
    env = {}

    if input_name is None:
        input_name = model.graph.input[0].name
    env[input_name] = (x_l, x_u)

    # convenience: store constants that appear as node inputs
    def get_bounds(name):
        if name in env:
            return env[name]
        if name in init:
            t = init[name]
            return t, t
        raise KeyError(f"Missing tensor in env/initializers: {name}")

    # run graph
    for node in model.graph.node:
        op = node.op_type
        ins = list(node.input)
        outs = list(node.output)
        if op == "Conv":
            l, u = get_bounds(ins[0])
            W = init[ins[1]]
            b = init[ins[2]] if len(ins) > 2 and ins[2] in init else None

            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            stride = tuple(attrs.get("strides", [1, 1]))
            pads = attrs.get("pads", [0, 0, 0, 0])
            # ONNX pads: [pad_top, pad_left, pad_bottom, pad_right]
            padding = (pads[0], pads[1])
            dilation = tuple(attrs.get("dilations", [1, 1]))
            groups = int(attrs.get("group", 1))

            lo, up = _ibp_conv2d(l, u, W, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
            env[outs[0]] = (lo, up)

        elif op == "Relu":
            l, u = get_bounds(ins[0])
            env[outs[0]] = _ibp_relu(l, u)

        elif op == "Gemm":
            # y = alpha*A*B + beta*C, with trans options
            A_l, A_u = get_bounds(ins[0])
            B = init[ins[1]]
            C = init[ins[2]] if len(ins) > 2 and ins[2] in init else None

            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            transA = int(attrs.get("transA", 0))
            transB = int(attrs.get("transB", 0))

            A_l2 = A_l.t() if transA else A_l
            A_u2 = A_u.t() if transA else A_u
            B2 = B.t() if transB else B

            # In many ONNX exports, Gemm uses B shaped (in, out). Our _ibp_linear expects W shaped (out, in).
            # So treat B2 as (in, out) and set W = B2.t() (out, in)
            W = (B2.t() * alpha)
            b = (C * beta) if C is not None else None

            lo, up = _ibp_linear(A_l2, A_u2, W, b)
            env[outs[0]] = (lo, up)

        elif op == "MatMul":
            # handle later if followed by Add; we still bound it here
            A_l, A_u = get_bounds(ins[0])
            B_l, B_u = get_bounds(ins[1])
            # If B is constant, we can do linear IBP; otherwise fallback is unsupported here.
            if not torch.allclose(B_l, B_u):
                raise NotImplementedError("IBP MatMul with non-constant right operand not implemented.")
            B = B_l
            # assume A is (N, in), B is (in, out)
            W = B.t()  # (out, in)
            lo, up = _ibp_linear(A_l, A_u, W, b=None)
            env[outs[0]] = (lo, up)

        elif op == "Add":
            a_l, a_u = get_bounds(ins[0])
            b_l, b_u = get_bounds(ins[1])
            env[outs[0]] = (a_l + b_l, a_u + b_u)

        elif op in ("Flatten", "Reshape"):
            l, u = get_bounds(ins[0])
            # Flatten: keep batch dim, flatten the rest
            if op == "Flatten":
                env[outs[0]] = (l.view(l.shape[0], -1), u.view(u.shape[0], -1))
            else:
                # Reshape: second input is shape tensor (usually constant)
                shape_l, shape_u = get_bounds(ins[1])
                shape = shape_l.to(torch.int64).cpu().numpy().tolist()
                env[outs[0]] = (l.reshape(shape), u.reshape(shape))
        elif op == "Transpose":
            l, u = get_bounds(ins[0])

            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            perm = attrs.get("perm", None)

            if perm is None:
                # ONNX default is reversing dimensions
                perm = list(range(l.dim()))[::-1]

            # torch expects a tuple
            perm = tuple(int(p) for p in perm)

            env[outs[0]] = (l.permute(perm).contiguous(),
                            u.permute(perm).contiguous())

        else:
            raise NotImplementedError(f"ONNX op not supported in this minimal IBP: {op}")

    if output_name is None:
        output_name = model.graph.output[0].name

    return env[output_name]


# --------- traversal order (Algorithm 1) ---------

def traversal_order_ibp(
    onnx_path,
    image,          # numpy: (H,W,C) or (H,W) or already (C,H,W)
    epsilon,
    label=None,     # if None, uses argmax on nominal logits
    input_min=0.0,
    input_max=1.0,
    channels_first=False,
):
    """
    Implements Algorithm 1: build m boxes (only feature i varies), run IBP once in batch, sort by lower_c desc.
    Returns: (sorted_indices_desc, per_feature_lower_bounds)
    """
    img = np.asarray(image).astype(np.float32)
    if img.ndim == 2:
        img = img[..., None]  # (H,W,1)

    H, W, C = img.shape
    m = H * W  # ranking over pixels (matches your VeriX indexing) :contentReference[oaicite:4]{index=4}

    flat = img.reshape(-1, C)  # (m, C)
    # batch of size m: each element is a full image, but only one pixel has interval
    base = np.kron(np.ones((m, 1, 1), dtype=np.float32), flat[None, :, :])  # (m, m, C)
    lower = base.copy()
    upper = base.copy()

    # set per-feature interval (pixel i)
    for i in range(m):
        lo = np.maximum(input_min, flat[i] - epsilon)
        up = np.minimum(input_max, flat[i] + epsilon)
        lower[i, i, :] = lo
        upper[i, i, :] = up

    # reshape to image batch
    lower = lower.reshape(m, H, W, C)
    upper = upper.reshape(m, H, W, C)

    # ONNX CNNs typically expect NCHW; adjust if needed
    if channels_first:
        lower_t = _as_torch(np.moveaxis(lower, -1, 1))  # (m,C,H,W)
        upper_t = _as_torch(np.moveaxis(upper, -1, 1))
        nominal = _as_torch(np.moveaxis(img[None, ...], -1, 1))
    else:
        # If your ONNX expects NHWC, this path might work, but most exported conv nets are NCHW.
        lower_t = _as_torch(lower)
        upper_t = _as_torch(upper)
        nominal = _as_torch(img[None, ...])

    # get label from nominal forward pass using IBP on a point-box
    if label is None:
        pt_l, pt_u = nominal, nominal
        l0, u0 = ibp_onnx_logits_bounds(onnx_path, pt_l, pt_u)
        logits = l0[0].detach().cpu().numpy()
        label = int(logits.argmax())

    logits_l, logits_u = ibp_onnx_logits_bounds(onnx_path, lower_t, upper_t)
    per_feature_lower = logits_l[:, label].detach().cpu().numpy()  # (m,)

    sorted_idx = per_feature_lower.argsort()[::-1]  # descending (paper + your code) 
    return sorted_idx, per_feature_lower.reshape(H, W)

# Example usage:
# from verix_traversal_ibp_onnx import traversal_order_ibp

# sorted_idx, sensitivity_map = traversal_order_ibp(
#     onnx_path=model_path + ".onnx",
#     image=self.image,          # (H,W,C)
#     epsilon=epsilon,
#     label=self.label,
#     channels_first=True        # most ONNX conv nets
# )

# self.inputVars = sorted_idx
# self.sensitivity = sensitivity_map
