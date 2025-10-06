# test_dependency.py
#
# Simple, assertion-style tests for the Stage 1 incremental API.
# Mirrors the style of test_network.py (no heavy mocking).
#
# Focus:
#   - incrementalSolve exists on MarabouNetwork
#   - sanity/error paths fire early (incremental_mode required, propertyFilename rejected)
#   - options exposure of _incremental and DnC guard via _snc
#   - getIncrementalInputQueries requirement (until implemented)
#
# TODOs are placeholders for later stages (once per-point IPQs & analyzer are wired).

from maraboupy import Marabou
from maraboupy import MarabouCore
import os
import numpy as np
import pytest

NETWORK_ONNX_FOLDER = "../../resources/onnx/"

def loadNetworkInONNX(filename, inputNames=None, outputName=None):
    # Load network in onnx relative to this file's location (same style as test_network.py)
    filename = os.path.join(os.path.dirname(__file__), NETWORK_ONNX_FOLDER, filename)
    return Marabou.read_onnx(filename, inputNames, outputName)


# ----------- Existence & basic flags -----------

def test_incremental_method_exists():
    # Ensure the Python class exposes incrementalSolve
    from maraboupy.MarabouNetwork import MarabouNetwork
    assert hasattr(MarabouNetwork, "incrementalSolve")


def test_options_has_incremental_flag_pybind():
    # Options wrapper exposes _incremental and it is settable
    opt = MarabouCore.Options()
    assert hasattr(opt, "_incremental")
    assert opt._incremental in (True, False)
    opt._incremental = True
    assert opt._incremental is True


# ----------- Error-path sanity checks (no analyzer required) -----------

def test_incremental_mode_required():
    # incrementalSolve should raise if incremental_mode is not enabled
    # Use any tiny ONNX in the resources folder; the call should fail before solving.
    network = loadNetworkInONNX("fc_2-2-3.onnx")
    threw = False
    try:
        network.incrementalSolve(verbose=False)
    except RuntimeError:
        threw = True
    assert threw


def test_property_filename_rejected():
    # Non-empty propertyFilename should raise (Stage 1 behavior)
    network = loadNetworkInONNX("fc_2-2-3.onnx")
    # Simulate addRobustnessBatch having been called (we only need the flag for this check)
    network.incremental_mode = True
    threw = False
    try:
        network.incrementalSolve(propertyFilename="foo.vnnlib", verbose=False)
    except NotImplementedError:
        threw = True
    assert threw


def test_dnc_true_rejected():
    # If DnC (_snc) is True, incrementalSolve should raise before attempting to build IPQs
    network = loadNetworkInONNX("fc_2-2-3.onnx")
    network.incremental_mode = True  # simulate addRobustnessBatch
    opt = Marabou.createOptions()
    # If the binding has _snc (it should), flip it on to trigger the guard
    if hasattr(opt, "_snc"):
        opt._snc = True
        threw = False
        try:
            network.incrementalSolve(options=opt, verbose=False)
        except ValueError:
            threw = True
        assert threw
    else:
        # If for some reason _snc is not present in this build, at least assert the method runs to the next guard
        passed = False
        try:
            network.incrementalSolve(options=opt, verbose=False)
        except RuntimeError:
            # expected later guard (no IPQs)
            passed = True
        assert passed


def test_incremental_batch_required():
    """
    If incremental_mode is True but addRobustnessBatch was never called (or batch empty),
    incrementalSolve should fail early via getIncrementalInputQueries().
    """
    network = loadNetworkInONNX("fc_2-2-3.onnx")
    network.incremental_mode = True  # simulate someone flipping the flag only

    opt = Marabou.createOptions()
    if hasattr(opt, "_snc"):
        opt._snc = False

    threw = False
    try:
        network.incrementalSolve(options=opt, verbose=False)
    except RuntimeError as e:
        msg = str(e)
        threw = ("Incremental batch is empty" in msg) or ("addRobustnessBatch" in msg)
    assert threw


def test_no_ipqs_error():
    """
    If getIncrementalInputQueries() returns an empty list (e.g., builder bug),
    incrementalSolve should surface the 'returned no IPQs' error.
    """
    network = loadNetworkInONNX("fc_2-2-3.onnx")

    # Properly simulate incremental setup (so we pass the early batch check)
    network.addRobustnessBatch(points=[[0.0, 0.0]], epsilon=0.01, targetLabel=None)

    # Monkeypatch the builder to return an empty list
    original = network.getIncrementalInputQueries
    network.getIncrementalInputQueries = lambda: []

    opt = Marabou.createOptions()
    if hasattr(opt, "_snc"):
        opt._snc = False

    threw = False
    try:
        network.incrementalSolve(options=opt, verbose=False)
    except RuntimeError as e:
        threw = ("no IPQs" in str(e)) or ("no IPQs" in repr(e))
    finally:
        # restore
        network.getIncrementalInputQueries = original

    assert threw


def test_addRobustnessBatch_sets_flags_and_disjunction():
    """
    - addRobustnessBatch should set incremental flags/fields
    - When targetLabel is provided, exactly one disjunction should be added
      with (#classes - 1) disjuncts of the form: y_t - y_k <= margin
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    # The toy net has 3 outputs; pick target label 1
    points = [[0.5, 0.2]]  # model has 2 inputs
    epsilon = 0.1
    target = 1
    margin = 0.0

    net.addRobustnessBatch(points, epsilon, targetLabel=target, margin=margin)

    assert net.incremental_mode is True
    assert net.incremental_points == points
    assert net.incremental_epsilon == epsilon

    # A single disjunction was queued
    assert len(net.disjunctionList) == 1
    disj = net.disjunctionList[0]
    # Expect (#classes - 1) = 2 disjuncts
    assert len(disj) == 2

    # Each disjunct should be a single LE inequality (+1)*y_t + (-1)*y_k <= margin
    # We can't directly read y_t/y_k ids here, but we can check Equation types and addends count.
    for d in disj:
        assert len(d) == 1
        eq = d[0]
        # LE inequality:
        assert eq.EquationType == MarabouCore.Equation.LE
        # Two addends: (+1)*y_t and (-1)*y_k
        assert len(eq.addendList) == 2
        # Right-hand side is the margin
        assert abs(eq.scalar - margin) < 1e-9

# TODO:
# def test_clearProperty_resets_incremental_state():
#     """
#     clearProperty should reset incremental flags and stored batch.
#     """
#     net = loadNetworkInONNX("fc_2-2-3.onnx")
#     net.addRobustnessBatch([[0.0, 0.0]], 0.05, targetLabel=0)
#     assert net.incremental_mode
#     assert len(net.incremental_points) == 1

#     net.clearProperty()
#     assert net.incremental_mode is False
#     assert net.incremental_points == []
#     assert net.incremental_epsilon is None


def test_getIncrementalInputQueries_bounds_and_count():
    """
    getIncrementalInputQueries should:
    - return one IPQ per point
    - set each input var to [max(p[i]-eps,0), min(p[i]+eps,1)] (current clamp behavior)
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    # 2-D input; choose a point near [0,1] to exercise clamping
    pts = [
        [0.02, 0.98],   # clamps lower/upper for both ends
        [0.5, 0.5],
    ]
    eps = 0.1
    net.addRobustnessBatch(pts, eps, targetLabel=0)

    ipqs = net.getIncrementalInputQueries()
    assert len(ipqs) == len(pts)

    # Flatten input var order like getInputQuery
    flat_in = [int(v) for arr in net.inputVars for v in arr.flatten()]

    # Check bounds for the first two points
    for p_idx, p in enumerate(pts):
        ipq = ipqs[p_idx]
        for i, var in enumerate(flat_in):
            lb = ipq.getLowerBound(var)
            ub = ipq.getUpperBound(var)
            exp_lb = max(float(p[i] - eps), 0.0)
            exp_ub = min(float(p[i] + eps), 1.0)
            assert abs(lb - exp_lb) < 1e-9
            assert abs(ub - exp_ub) < 1e-9


def test_getIncrementalInputQueries_length_mismatch_asserts():
    """
    If a point does not match #inputs, a ValueError should be raised by addRobustnessBatch.
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    # This network expects 2 inputs; pass 3 to trigger the error
    with pytest.raises(ValueError, match="expected 2"):
        net.addRobustnessBatch([[0.1, 0.2, 0.3]], 0.01, targetLabel=0)


def test_incrementalSolve_per_point_filename_and_lengths(monkeypatch, capsys):
    """
    - incrementalSolve should call Core.solve once per IPQ and append per-point filename suffix
      when a base filename is provided.
    - The returned lists (exitCodes, valsList, statsList) should match the batch size.
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    pts = [[0.0, 0.0], [0.1, 0.5], [0.2, 0.2]]
    net.addRobustnessBatch(pts, 0.05, targetLabel=0)

    # Spy on redirects that MarabouCore.solve receives
    redirects = []
    def fake_solve(ipq, options, redirect):
        redirects.append(redirect)
        # Return a minimal shape-compatible triplet
        return ("UNKNOWN", {}, None)

    monkeypatch.setattr(MarabouCore, "solve", fake_solve)

    opt = Marabou.createOptions(verbosity=0)

    base = "tmp.out"
    exitCodes, valsList, statsList = net.incrementalSolve(filename=base, verbose=False, options=opt)

    # Batch size respected
    assert len(exitCodes) == len(pts)
    assert len(valsList) == len(pts)
    assert len(statsList) == len(pts)

    # Filenames are suffixed .pt000, .pt001, .pt002
    assert redirects == [f"{base}.pt000", f"{base}.pt001", f"{base}.pt002"]


def _flat_inputs(net):
    return [int(v) for arr in net.inputVars for v in arr.flatten()]

def test_addRobustnessBatch_raises_if_points_outside_minmax():
    """
    When custom input_min/max are provided, all points must lie within them.
    Violations should raise a ValueError.
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    pts = [[-0.5, 0.2], [0.5, 1.5]]  # first below min, second above max

    with pytest.raises(ValueError, match="outside declared bounds"):
        net.addRobustnessBatch(
            pts, epsilon=0.05, targetLabel=0,
            input_min=0.0, input_max=1.0
        )



def test_bounds_scalar_minmax():
    """
    Scalar min/max should broadcast to all input dimensions.
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    pts = [[0.2, 0.8]]
    eps = 0.15

    # non-default scalar range
    inp_min = 0.1
    inp_max = 0.9

    net.addRobustnessBatch(pts, eps, targetLabel=1, input_min=inp_min, input_max=inp_max)
    ipqs = net.getIncrementalInputQueries()
    assert len(ipqs) == 1

    ipq = ipqs[0]
    flat_in = _flat_inputs(net)

    # lb = max(p-eps, 0.1), ub = min(p+eps, 0.9)
    p = pts[0]
    exp_lbs = [max(p[0] - eps, inp_min), max(p[1] - eps, inp_min)]
    exp_ubs = [min(p[0] + eps, inp_max), min(p[1] + eps, inp_max)]

    for v, lb, ub in zip(flat_in, exp_lbs, exp_ubs):
        assert abs(ipq.getLowerBound(v) - lb) < 1e-9
        assert abs(ipq.getUpperBound(v) - ub) < 1e-9


def test_bounds_vector_minmax_asymmetric():
    """
    Per-dimension min/max vectors should be respected.
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    pts = [[0.0, 1.0]]
    eps = 0.2

    inp_min = [-0.5, 0.2]
    inp_max = [0.3, 1.7]

    net.addRobustnessBatch(pts, eps, targetLabel=2, input_min=inp_min, input_max=inp_max)
    ipqs = net.getIncrementalInputQueries()
    assert len(ipqs) == 1

    ipq = ipqs[0]
    flat_in = _flat_inputs(net)

    p = pts[0]
    exp_lbs = [max(p[0] - eps, inp_min[0]), max(p[1] - eps, inp_min[1])]
    exp_ubs = [min(p[0] + eps, inp_max[0]), min(p[1] + eps, inp_max[1])]

    for v, lb, ub in zip(flat_in, exp_lbs, exp_ubs):
        assert abs(ipq.getLowerBound(v) - lb) < 1e-9
        assert abs(ipq.getUpperBound(v) - ub) < 1e-9


def test_bounds_bad_length_raises():
    """
    input_min/input_max vectors must match #inputs; otherwise an error is raised.
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    pts = [[0.1, 0.2]]
    eps = 0.05

    # too many entries (network has 2 inputs)
    with pytest.raises((ValueError, AssertionError)):
        net.addRobustnessBatch(pts, eps, targetLabel=0, input_min=[0.0, 0.0, 0.0], input_max=[1.0, 1.0, 1.0])

    # mismatched lengths
    with pytest.raises((ValueError, AssertionError)):
        net.addRobustnessBatch(pts, eps, targetLabel=0, input_min=[0.0], input_max=[1.0, 1.0])


def test_bounds_min_gt_max_raises():
    """
    If any min > max, an error should be raised.
    """
    net = loadNetworkInONNX("fc_2-2-3.onnx")
    pts = [[0.1, 0.2]]
    eps = 0.05

    # scalar case: min > max
    with pytest.raises((ValueError, AssertionError)):
        net.addRobustnessBatch(pts, eps, targetLabel=0, input_min=1.0, input_max=0.0)

    # vector case: one dim has min > max
    with pytest.raises((ValueError, AssertionError)):
        net.addRobustnessBatch(pts, eps, targetLabel=0, input_min=[-1.0, 0.8], input_max=[0.5, 0.7])



# ----------- TODOs for later stages -----------

def test_per_point_filename_suffix_TODO():
    # TODO: After getIncrementalInputQueries is implemented and returns >1 IPQ,
    # call incrementalSolve(filename="run") and assert that files run.pt000, run.pt001, ...
    # are used (or that solve() receives those redirect strings).
    assert True  # placeholder


def test_returns_match_batch_size_TODO():
    # TODO: After IPQ builder exists, ensure returned exitCodes/valsList/statsList
    # lengths equal the batch size.
    assert True  # placeholder


def test_verbose_print_shape_TODO():
    # TODO: With a tiny batch where SAT is forced for at least one IPQ,
    # ensure incrementalSolve(verbose=True) prints per-point headers and input/output lines.
    assert True  # placeholder

def test_shared_output_property_added_once():
    from maraboupy import Marabou
    net = Marabou.read_onnx(os.path.join(os.path.dirname(__file__), "../../resources/onnx/fc_2-2-3.onnx"))
    # one dummy point with correct length
    n_inputs = net.inputVars[0][0].size
    net.addRobustnessBatch(points=[[0.0]*n_inputs], epsilon=0.01, targetLabel=0, margin=0.0)

    ipqs = net.getIncrementalInputQueries()
    # There should be exactly one IPQ
    assert len(ipqs) == 1
    # The inequalities we just added live in additionalEquList and get copied into the IPQ,
    # so solve() should "see" them. We don’t solve here; just ensure no exceptions thrown
    # and IPQ constructed successfully.
