'''
Top contributors (to current version):
    - Christopher Lazarus
    - Shantanu Thakoor
    - Andrew Wu
    - Kyle Julian
    - Teruhiro Tagomori
    - Min Wu

This file is part of the Marabou project.
Copyright (c) 2017-2024 by the authors listed in the file AUTHORS
in the top-level source directory) and their institutional affiliations.
All rights reserved. See the file COPYING in the top-level source
directory for licensing information.
'''

from maraboupy import MarabouCore
from maraboupy import MarabouUtils
from maraboupy.MarabouPythonic import *
from abc import ABC

class InputQueryBuilder(ABC):
    """
    Abstract class for building up an input query.
    Should eventually be implemented by a renamed `NetworkParser.cpp` on the C++ side.
    """
    def __init__(self):
        """
        Constructs a MarabouNetwork object and calls function to initialize
        """
        self.clear()

    def clear(self):
        """Reset values to represent empty network
        """
        self.numVars = 0
        self.equList = []
        self.additionalEquList = [] # used to store user defined equations
        self.reluList = []
        self.leakyReluList = []
        self.sigmoidList = []
        self.maxList = []
        self.softmaxList = []
        self.bilinearList = []
        self.absList = []
        self.signList = []
        self.disjunctionList = []
        self.lowerBounds = dict()
        self.upperBounds = dict()
        self.inputVars = []
        self.outputVars = []
        # Incrementality variables
        self.incremental_mode = False
        self.incremental_input_lbs = None
        self.incremental_input_ubs = None
        self.incremental_input_min = None   # per-dim mins or scalar
        self.incremental_input_max = None   # per-dim maxs or scalar

    def clearProperty(self):
        """Clear the lower bounds and upper bounds map, and the self.additionEquList
            and incrementality variables if deactivate incrementality if used.
        """
        self.lowerBounds.clear()
        self.upperBounds.clear()
        self.additionalEquList.clear()
        if self.incremental_mode: 
            self.incremental_mode = False
            self.incremental_input_lbs = None
            self.incremental_input_ubs = None

    def getNewVariable(self):
        """Function to create a new variable

        Returns:
            (int): New variable number

        :meta private:
        """
        self.numVars += 1
        return self.numVars - 1

    def addEquation(self, x, isProperty=False):
        """Function to add new equation to the network

        Args:
            x (:class:`~maraboupy.MarabouUtils.Equation`): New equation to add
            isProperty (bool): If true, this constraint can be removed later by clearProperty() method
        """
        if isProperty:
            self.additionalEquList += [x]
        else:
            self.equList += [x]

    def addRobustnessBatch(self, points, epsilons, targetLabel=None, margin=0.00001,
                        input_min=0.0, input_max=1.0):
        """
        Add a batch for incremental robustness and (optionally) encode the negated
        robustness property as a single disjunction:  OR_{k!=t} ( y_t - y_k <= -margin ).
        i.e., some other class k is at least `margin` larger than y_t.

        Args:
            points (list[list[float]] | np.ndarray):
                Base inputs (flattened), shape (num_points, num_inputs).
            epsilons (list[float] | np.ndarray):
                Per-point L-infinity radius around each point, shape (num_points,).
            targetLabel (int|None):
                If provided, encodes OR_{k!=t}(y_t - y_k <= margin).
            margin (float):
                Non-negative class-separation margin.
            input_min (float | list[float] | np.ndarray):
                Per-dimension or scalar lower limits on the *allowed* input range.
            input_max (float | list[float] | np.ndarray):
                Per-dimension or scalar upper limits on the *allowed* input range.
        """
        import numpy as np

        self.incremental_mode = True

        # Normalize points and epsilons
        points = np.asarray(points, dtype=float)
        epsilons = np.asarray(epsilons, dtype=float)

        assert points.ndim == 2, f"points must be 2D (num_points, num_inputs), got shape {points.shape}"
        assert epsilons.ndim == 1, f"epsilons must be 1D (num_points,), got shape {epsilons.shape}"
        assert epsilons.shape[0] == points.shape[0], \
            f"len(epsilons) = {epsilons.shape[0]} must match num_points = {points.shape[0]}"

        num_points, point_dim = points.shape

        # Determine number of input dims from the network
        flat_input_vars = []
        for inputVarArray in self.inputVars:
            for inputVar in inputVarArray.flatten():
                flat_input_vars.append(int(inputVar))
        num_inputs = len(flat_input_vars)
        if num_inputs == 0:
            raise RuntimeError("No input variables found; cannot set batch.")

        if point_dim != num_inputs:
            raise ValueError(
                f"Each point must have length {num_inputs}, got {point_dim}."
            )

        # Broadcast/validate input_min / input_max to per-dim arrays
        def _as_per_dim(arr_or_scalar, name):
            arr = np.asarray(arr_or_scalar, dtype=float)
            if arr.ndim == 0:                # scalar -> broadcast
                return np.full((num_inputs,), float(arr), dtype=float)
            if arr.shape == (num_inputs,):   # already per-dim
                return arr.astype(float)
            raise ValueError(f"{name} must be scalar or length {num_inputs}, got shape {arr.shape}")

        mins = _as_per_dim(input_min, "input_min")
        maxs = _as_per_dim(input_max, "input_max")
        if not np.all(mins <= maxs + 0.0):
            raise ValueError("input_min must be <= input_max elementwise")

        # --- Validate that all points lie within the provided [min, max] bounds
        for idx, p in enumerate(points):
            if len(p) != num_inputs:
                raise ValueError(f"Point #{idx} has length {len(p)}, expected {num_inputs}.")
            if np.any(p < mins - 1e-9) or np.any(p > maxs + 1e-9):
                raise ValueError(
                    f"Point #{idx} contains values outside declared bounds: "
                    f"min(p)={np.min(p):.5f}, max(p)={np.max(p):.5f}, "
                    f"expected in [{np.min(mins):.5f}, {np.max(maxs):.5f}]"
                )

        # These are the global allowed domain bounds (not per-query boxes)
        self.incremental_input_min = mins
        self.incremental_input_max = maxs

        # --- Per-query L_inf boxes: lb/ub for each point
        # Clip to [mins, maxs] to respect global domain
        lbs = points - epsilons[:, None]
        ubs = points + epsilons[:, None]
        lbs = np.maximum(lbs, mins)   # shape (num_points, num_inputs)
        ubs = np.minimum(ubs, maxs)

        # Store them for the incremental analyzer
        self.incremental_input_lbs = lbs
        self.incremental_input_ubs = ubs

        # Optionally encode the shared disjunction (negated robustness)
        if targetLabel is not None:
            flat_outputs = []
            for outputVarArray in self.outputVars:
                for outputVar in outputVarArray.flatten():
                    flat_outputs.append(int(outputVar))

            assert len(flat_outputs) > 0, "No output variables found when adding class property"
            assert 0 <= targetLabel < len(flat_outputs), \
                f"targetLabel {targetLabel} out of range (num outputs = {len(flat_outputs)})"
            assert margin >= 0.0, "margin must be non-negative"

            y_t = flat_outputs[targetLabel]

            # Build disjunction:  OR_k  ( y_t - y_k <= margin )
            # Encode each disjunct as: (+1)*y_t + (-1)*y_k <= margin
            disjuncts = []
            for k, y_k in enumerate(flat_outputs):
                if k == targetLabel:
                    continue
                eq = MarabouUtils.Equation(MarabouCore.Equation.LE)
                eq.addAddend(1.0, y_t)
                eq.addAddend(-1.0, y_k)
                eq.setScalar(float(-margin))
                # Each disjunct is a list of equations; here each disjunct has exactly one inequality
                disjuncts.append([eq])

            # Add a single DisjunctionConstraint with all disjuncts
            self.addDisjunctionConstraint(disjuncts)
        else:
            print("[DEBUG] targetLabel not assigned, no output constraints were applied.")

        print(
            f"[DEBUG] addRobustnessBatch: n_points={num_points}, "
            f"epsilons=[min={epsilons.min():.3g}, max={epsilons.max():.3g}], "
            f"targetLabel={targetLabel}, margin={-margin}, "
            f"[min/max]=[{np.min(self.incremental_input_min):.3g},"
            f"{np.max(self.incremental_input_max):.3g}]"
        )


    def setLowerBound(self, x, v):
        """Function to set lower bound for variable

        Args:
            x (int): Variable number to set
            v (float): Value representing lower bound
        """
        self.lowerBounds[x]=v

    def setUpperBound(self, x, v):
        """Function to set upper bound for variable

        Args:
            x (int): Variable number to set
            v (float): Value representing upper bound
        """
        self.upperBounds[x]=v

    def addRelu(self, v1, v2):
        """Function to add a new Relu constraint

        Args:
            v1 (int): Variable representing input of Relu
            v2 (int): Variable representing output of Relu
        """
        self.reluList += [(v1, v2)]

    def addLeakyRelu(self, v1, v2, slope):
        """Function to add a new Leaky Relu constraint

        Args:
            v1 (int): Variable representing input of Leaky Relu
            v2 (int): Variable representing output of Leaky Relu
            slope (float): Shope of the Leaky ReLU
        """
        self.leakyReluList += [(v1, v2, slope)]

    def addBilinear(self, v1, v2, v3):
        """Function to add a bilinear constraint to the network
        Args:
            v1 (int): Variable representing input1 of Bilinear
            v2 (int): Variable representing input2 of Bilinear
            v3 (int): Variable representing output of Bilinear
        """
        self.bilinearList += [(v1, v2, v3)]

    def addSigmoid(self, v1, v2):
        """Function to add a new Sigmoid constraint

        Args:
            v1 (int): Variable representing input of Sigmoid
            v2 (int): Variable representing output of Sigmoid
        """
        self.sigmoidList += [(v1, v2)]

    def addMaxConstraint(self, elements, v):
        """Function to add a new Max constraint

        Args:
            elements (set of int): Variable representing input to max constraint
            v (int): Variable representing output of max constraint
        """
        self.maxList += [(elements, v)]

    def addSoftmaxConstraint(self, inputs, outputs):
        """Function to add a new softmax constraint

        Args:
            inputs (set of int): Variable representing input to max constraint
            outputs (set of int): Variables representing outputs of max constraint
        """
        self.softmaxList += [(inputs, outputs)]

    def addAbsConstraint(self, b, f):
        """Function to add a new Abs constraint

        Args:
            b (int): Variable representing input of the Abs constraint
            f (int): Variable representing output of the Abs constraint
        """
        self.absList += [(b, f)]

    def addSignConstraint(self, b, f):
        """Function to add a new Sign constraint

        Args:
            b (int): Variable representing input of Sign
            f (int): Variable representing output of Sign
        """
        self.signList += [(b, f)]

    def addDisjunctionConstraint(self, disjuncts):
        """Function to add a new Disjunction constraint

        Args:
            disjuncts (list of list of Equations): Each inner list represents a disjunct
        """
        self.disjunctionList.append(disjuncts)

    def lowerBoundExists(self, x):
        """Function to check whether lower bound for a variable is known

        Args:
            x (int): Variable to check
        """
        return x in self.lowerBounds

    def upperBoundExists(self, x):
        """Function to check whether upper bound for a variable is known

        Args:
            x (int): Variable to check
        """
        return x in self.upperBounds

    def addEquality(self, vars, coeffs, scalar, isProperty=False):
        """Function to add equality constraint to network

        .. math::
            \sum_i vars_i * coeffs_i = scalar

        Args:
            vars (list of int): Variable numbers
            coeffs (list of float): Coefficients
            scalar (float): Right hand side constant of equation
            isProperty (bool): If true, this constraint can be removed later by clearProperty() method
        """
        assert len(vars)==len(coeffs)
        e = MarabouUtils.Equation()
        for i in range(len(vars)):
            e.addAddend(coeffs[i], vars[i])
        e.setScalar(scalar)
        self.addEquation(e, isProperty)

    def addInequality(self, vars, coeffs, scalar, isProperty=False):
        """Function to add inequality constraint to network

        .. math::
            \sum_i vars_i * coeffs_i \le scalar

        Args:
            vars (list of int): Variable numbers
            coeffs (list of float): Coefficients
            scalar (float): Right hand side constant of inequality
            isProperty (bool): If true, this constraint can be removed later by clearProperty() method
        """
        assert len(vars)==len(coeffs)
        e = MarabouUtils.Equation(MarabouCore.Equation.LE)
        for i in range(len(vars)):
            e.addAddend(coeffs[i], vars[i])
        e.setScalar(scalar)
        self.addEquation(e, isProperty)

    def addConstraint(self, constraint: VarConstraint):
        """
        Support the Pythonic API to add constraints to the neurons in the Marabou network.

        :param constraint: an instance of the VarConstraint class, which comprises various neuron constraints.
        :return: delegate various constraints into lower/upper bounds and equality/inequality.
        """
        vars = list(constraint.combination.varCoeffs)
        coeffs = [constraint.combination.varCoeffs[i] for i in vars]
        if constraint.lowerBound is not None:
            self.setLowerBound(vars[0], constraint.lowerBound)
        elif constraint.upperBound is not None:
            self.setUpperBound(vars[0], constraint.upperBound)
        else:
            if constraint.isEquality:
                self.addEquality(vars, coeffs, - constraint.combination.scalar)
            else:
                self.addInequality(vars, coeffs, - constraint.combination.scalar)

    def getInputQuery(self):
        """Constructs the `InputQuery` object from the current set of constraints.

        Returns:
            :class:`~maraboupy.MarabouCore.InputQuery`
        """
        ipq = MarabouCore.InputQuery()
        ipq.setNumberOfVariables(self.numVars)

        i = 0
        for inputVarArray in self.inputVars:
            for inputVar in inputVarArray.flatten():
                ipq.markInputVariable(inputVar, i)
                i+=1

        i = 0
        for outputVarArray in self.outputVars:
            for outputVar in outputVarArray.flatten():
                ipq.markOutputVariable(outputVar, i)
                i+=1

        for e in self.equList:
            eq = MarabouCore.Equation(e.EquationType)
            for (c, v) in e.addendList:
                assert v < self.numVars
                eq.addAddend(c, v)
            eq.setScalar(e.scalar)
            ipq.addEquation(eq)

        for e in self.additionalEquList:
            eq = MarabouCore.Equation(e.EquationType)
            for (c, v) in e.addendList:
                assert v < self.numVars
                eq.addAddend(c, v)
            eq.setScalar(e.scalar)
            ipq.addEquation(eq)

        for r in self.reluList:
            assert r[1] < self.numVars and r[0] < self.numVars
            MarabouCore.addReluConstraint(ipq, r[0], r[1])

        for r in self.leakyReluList:
            assert r[1] < self.numVars and r[0] < self.numVars
            assert(r[2] > 0 and r[2] < 1)
            MarabouCore.addLeakyReluConstraint(ipq, r[0], r[1], r[2])

        for r in self.bilinearList:
            assert r[2] < self.numVars and r[1] < self.numVars and r[0] < self.numVars
            MarabouCore.addBilinearConstraint(ipq, r[0], r[1], r[2])

        for r in self.sigmoidList:
            assert r[1] < self.numVars and r[0] < self.numVars
            MarabouCore.addSigmoidConstraint(ipq, r[0], r[1])

        for m in self.maxList:
            assert m[1] < self.numVars
            for e in m[0]:
                assert e < self.numVars
            MarabouCore.addMaxConstraint(ipq, m[0], m[1])

        for m in self.softmaxList:
            for e in m[1]:
                assert e < self.numVars
            for e in m[0]:
                assert e < self.numVars
            MarabouCore.addSoftmaxConstraint(ipq, m[0], m[1])

        for b, f in self.absList:
            MarabouCore.addAbsConstraint(ipq, b, f)

        for b, f in self.signList:
            MarabouCore.addSignConstraint(ipq, b, f)

        for disjunction in self.disjunctionList:
            converted_disjunction = []
            for disjunct in disjunction:
                converted_disjunct = []
                for e in disjunct:
                    eq = MarabouCore.Equation(e.EquationType)
                    for (c, v) in e.addendList:
                        assert v < self.numVars
                        eq.addAddend(c, v)
                    eq.setScalar(e.scalar)
                    converted_disjunct.append(eq)
                converted_disjunction.append(converted_disjunct)
            MarabouCore.addDisjunctionConstraint(ipq, converted_disjunction)

        for l in self.lowerBounds:
            assert l < self.numVars
            ipq.setLowerBound(l, self.lowerBounds[l])

        for u in self.upperBounds:
            assert u < self.numVars
            ipq.setUpperBound(u, self.upperBounds[u])

        if self.incremental_mode:
            print(f"[DEBUG] Building InputQuery in incremental mode with {len(self.incremental_input_lbs)} points")
            # TODO: Pass points and epsilon to ipq once C++ side supports it

        return ipq

    def getCoveringInputBounds(self):
        """
        Compute covering input bounds over the whole batch, using the per-query
        L_inf boxes stored in `incremental_input_lbs` / `incremental_input_ubs`:

            cover_lb[i] = min_j incremental_input_lbs[j, i]
            cover_ub[i] = max_j incremental_input_ubs[j, i]

        Returns:
            (cover_lb, cover_ub): two Python lists of floats, length = #inputs

        Preconditions:
            - addRobustnessBatch(...) called
            - incremental_input_lbs / incremental_input_ubs are numpy arrays with
            shape (num_points, num_inputs)
        """

        if not self.incremental_mode:
            raise RuntimeError("getCoveringInputBounds called but incremental_mode is False")

        if self.incremental_input_lbs is None or self.incremental_input_ubs is None:
            raise RuntimeError(
                "incremental_input_lbs / incremental_input_ubs not initialized; "
                "call addRobustnessBatch first."
            )
        if self.incremental_input_min is None or self.incremental_input_max is None:
            raise RuntimeError(
                "input_min/input_max were not initialized; call addRobustnessBatch with min/max."
            )

        lbs = self.incremental_input_lbs
        ubs = self.incremental_input_ubs
        mins = self.incremental_input_min
        maxs = self.incremental_input_max

        # Assert we really have numpy arrays
        import numpy as np  # in case not already at top of file
        assert isinstance(lbs, np.ndarray), "incremental_input_lbs must be a numpy array"
        assert isinstance(ubs, np.ndarray), "incremental_input_ubs must be a numpy array"
        assert isinstance(mins, np.ndarray), "incremental_input_min must be a numpy array"
        assert isinstance(maxs, np.ndarray), "incremental_input_max must be a numpy array"

        assert lbs.ndim == 2 and ubs.ndim == 2, \
            f"incremental_input_lbs/ubs must be 2D (num_points, num_inputs), got {lbs.shape} and {ubs.shape}"
        assert lbs.shape == ubs.shape, \
            f"incremental_input_lbs and incremental_input_ubs shapes differ: {lbs.shape} vs {ubs.shape}"

        num_points, num_inputs = lbs.shape

        # Flatten inputs in the same order as getInputQuery()
        flat_input_vars = []
        for inputVarArray in self.inputVars:
            for inputVar in inputVarArray.flatten():
                flat_input_vars.append(int(inputVar))

        if num_inputs != len(flat_input_vars):
            raise RuntimeError(
                f"incremental_input_lbs/ubs have {num_inputs} dims, "
                f"but network has {len(flat_input_vars)} input vars."
            )

        cover_lb = []
        cover_ub = []
        for i in range(num_inputs):
            lb_i = float(np.min(lbs[:, i]))
            ub_i = float(np.max(ubs[:, i]))
            cover_lb.append(lb_i)
            cover_ub.append(ub_i)

        return cover_lb, cover_ub









    def getIncrementalInputQueries(self):
        """
        Construct a list of per-point InputQuery objects for incremental verification.

        Stage 1:
        - Requires that a batch was supplied via addRobustnessBatch(points, epsilons).
        - For each point, builds a fresh InputQuery via getInputQuery(), then
        sets the input variable bounds from the precomputed L-infinity boxes:
            lb_i = incremental_input_lbs[pt_idx, i]
            ub_i = incremental_input_ubs[pt_idx, i]
        - Returns: list of InputQuery objects, one per point.

        Preconditions:
            - self.incremental_mode is True
            - self.incremental_input_lbs / self.incremental_input_ubs are numpy arrays
            of shape (num_points, num_inputs)
        """
        import numpy as np

        # Preconditions
        if not self.incremental_mode:
            raise RuntimeError(
                "getIncrementalInputQueries called but incremental_mode is False. "
                "Call addRobustnessBatch(points, epsilons) first."
            )

        if self.incremental_input_lbs is None or self.incremental_input_ubs is None:
            raise RuntimeError(
                "Incremental batch is empty or not initialized. "
                "Ensure addRobustnessBatch(points, epsilons) was called with a non-empty list."
            )
        if self.incremental_input_min is None or self.incremental_input_max is None:
            raise RuntimeError("input_min/input_max were not initialized; call addRobustnessBatch with min/max.")

        lbs = self.incremental_input_lbs
        ubs = self.incremental_input_ubs

        # Assert numpy arrays
        assert isinstance(lbs, np.ndarray), "incremental_input_lbs must be a numpy array"
        assert isinstance(ubs, np.ndarray), "incremental_input_ubs must be a numpy array"

        assert lbs.ndim == 2 and ubs.ndim == 2, \
            f"incremental_input_lbs/ubs must be 2D (num_points, num_inputs), got {lbs.shape} and {ubs.shape}"
        assert lbs.shape == ubs.shape, \
            f"incremental_input_lbs and incremental_input_ubs shapes differ: {lbs.shape} vs {ubs.shape}"

        num_points, num_inputs = lbs.shape

        # Flatten input vars in the same order as getInputQuery()
        flat_input_vars = []
        for inputVarArray in self.inputVars:
            for inputVar in inputVarArray.flatten():
                flat_input_vars.append(int(inputVar))

        if num_inputs != len(flat_input_vars):
            raise RuntimeError(
                f"incremental_input_lbs/ubs have {num_inputs} dims, "
                f"but network has {len(flat_input_vars)} input vars."
            )

        # Build one IPQ per point
        ipqs = []
        print(f"[DEBUG] getIncrementalInputQueries: building {num_points} IPQs")

        for pt_idx in range(num_points):
            ipq = self.getInputQuery()  # includes current constraints, IO markings, etc.

            for i, var in enumerate(flat_input_vars):
                lb = float(lbs[pt_idx, i])
                ub = float(ubs[pt_idx, i])

                if lb > ub:
                    raise ValueError(
                        f"Invalid bounds at point {pt_idx}, dim {i}: [{lb}, {ub}]"
                    )

                ipq.setLowerBound(var, lb)
                ipq.setUpperBound(var, ub)

            ipqs.append(ipq)

        return ipqs





    def saveQuery(self, filename=""):
        """Serializes the inputQuery in the given filename

        Args:
            filename: (string) file to write serialized inputQuery
        """
        ipq = self.getInputQuery()
        MarabouCore.saveQuery(ipq, str(filename))

    def isEqualTo(self, network):
        """
        Add a comparison between two Marabou networks and all their attributes.

        :param network: the other Marabou network to be compared with.
        :return: True if these two networks and all their attributes are identical; False if not.
        """
        equivalence = True
        if self.numVars != network.numVars \
                or self.reluList != network.reluList \
                or self.sigmoidList != network.sigmoidList \
                or self.maxList != network.maxList \
                or self.absList != network.absList \
                or self.signList != network.signList \
                or self.disjunctionList != network.disjunctionList \
                or self.lowerBounds != network.lowerBounds \
                or self.upperBounds != network.upperBounds:
            equivalence = False
        for equation1, equation2 in zip(self.equList, network.equList):
            if not equation1.isEqualTo(equation2):
                equivalence = False
        for inputvars1, inputvars2 in zip(self.inputVars, network.inputVars):
            if (inputvars1.flatten() != inputvars2.flatten()).any():
                equivalence = False
        for outputVars1, outputVars2 in zip(self.outputVars, network.outputVars):
            if (outputVars1.flatten() != outputVars1.flatten()).any():
                equivalence = False
        return equivalence
