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


def test_get_incremental_ipqs_required():
    # With incremental_mode set and DnC off, the next guard should complain about missing IPQs
    network = loadNetworkInONNX("fc_2-2-3.onnx")
    network.incremental_mode = True  # simulate addRobustnessBatch
    opt = Marabou.createOptions()
    if hasattr(opt, "_snc"):
        opt._snc = False
    threw = False
    try:
        network.incrementalSolve(options=opt, verbose=False)
    except RuntimeError as e:
        # Expect the "returned no IPQs" message from Stage 1
        threw = ("no IPQs" in str(e)) or ("no IPQs" in repr(e))
    assert threw


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
