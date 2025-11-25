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

MarabouNetwork defines an abstract class that represents neural networks with piecewise linear constraints
'''

from maraboupy import MarabouCore
from maraboupy.parsers.InputQueryBuilder import InputQueryBuilder
import numpy as np


class MarabouNetwork(InputQueryBuilder):
    """Abstract class representing general Marabou network

    Attributes:
        numVars (int): Total number of variables to represent network
        equList (list of :class:`~maraboupy.MarabouUtils.Equation`): Network equations
        reluList (list of tuples): List of relu constraint tuples, where each tuple contains the backward and forward variables
        leakyReluList (list of tuples): List of leaky relu constraint tuples, where each tuple contains the backward and forward variables, and the slope
        sigmoidList (list of tuples): List of sigmoid constraint tuples, where each tuple contains the backward and forward variables
        maxList (list of tuples): List of max constraint tuples, where each tuple conatins the set of input variables and output variable
        absList (list of tuples): List of abs constraint tuples, where each tuple conatins the input variable and the output variable
        signList (list of tuples): List of sign constraint tuples, where each tuple conatins the input variable and the output variable
        lowerBounds (Dict[int, float]): Lower bounds of variables
        upperBounds (Dict[int, float]): Upper bounds of variables
        inputVars (list of numpy arrays): Input variables
        outputVars (list of numpy arrays): Output variables
    """
    def __init__(self):
        """
        Constructs a MarabouNetwork object and calls function to initialize
        """
        super().__init__()
        self.clear()

    def clearProperty(self):
        """Clear the lower bounds and upper bounds map, and the self.additionEquList
        """
        self.lowerBounds.clear()
        self.upperBounds.clear()
        self.additionalEquList.clear()

    def solve(self, filename="", verbose=True, options=None, propertyFilename=""):
        """Function to solve query represented by this network

        Args:
            filename (string): Path for redirecting output
            verbose (bool): If true, print out solution after solve finishes
            options (:class:`~maraboupy.MarabouCore.Options`): Object for specifying Marabou options, defaults to None
            propertyFilename(string): Path for property file

        Returns:
            (tuple): tuple containing:
                - exitCode (str): A string representing the exit code (sat/unsat/TIMEOUT/ERROR/UNKNOWN/QUIT_REQUESTED).
                - vals (Dict[int, float]): Empty dictionary if UNSAT, otherwise a dictionary of SATisfying values for variables
                - stats (:class:`~maraboupy.MarabouCore.Statistics`): A Statistics object to how Marabou performed
        """
        ipq = self.getInputQuery()
        if propertyFilename:
            MarabouCore.loadProperty(ipq, propertyFilename)
        if options == None:
            options = MarabouCore.Options()
        exitCode, vals, stats = MarabouCore.solve(ipq, options, str(filename))
        if verbose:
            print(exitCode)
            if exitCode == "sat":
                for j in range(len(self.inputVars)):
                    for i in range(self.inputVars[j].size):
                        print("input {} = {}".format(i, vals[self.inputVars[j].item(i)]))

                for j in range(len(self.outputVars)):
                    for i in range(self.outputVars[j].size):
                        print("output {} = {}".format(i, vals[self.outputVars[j].item(i)]))

        return [exitCode, vals, stats]
    
    def incrementalSolve(self, filename: str = "", verbose: bool = True,
                        options=None, propertyFilename: str = ""):
        """
        Function to solve a batch of robustness queries incrementally.

        This runs the solver once per input point in the previously supplied batch
        (via `addRobustnessBatch(points, epsilon)`), while reusing shared bounds
        and dependency information across the batch (incremental mode).

        Args:
            filename (string):
                Base path for redirecting output. If non-empty, a per-point suffix
                is appended: "<filename>.pt{idx}".
            verbose (bool):
                If True, prints per-point results after solving.
            options (:class:`~maraboupy.MarabouCore.Options`):
                Solver options. If None, a fresh Options() is created.
                Requirements for incremental mode:
                - options.incremental must be settable to True (set internally).
                - options.dnc must be False; if it is True, an error is raised.
            propertyFilename (string):
                Reserved for future use. For now, only the empty string "" is
                supported. If a non-empty string is passed, an error is raised.

        Returns:
            (tuple):
                - exitCodes (List[str]): One per point in the batch, each in
                {"sat","unsat","TIMEOUT","ERROR","UNKNOWN","QUIT_REQUESTED"}.
                - valsList (List[Dict[int, float]]): For each point, empty dict if
                UNSAT; otherwise satisfying assignment (var -> value).
                - statsList (List[:class:`~maraboupy.MarabouCore.Statistics`]):
                One Statistics object per point.

        Notes:
            - Expects `getIncrementalInputQueries()` to return per-point IPQs that
            all share the same (non-owning) DependencyAnalyzer pointer set on C++ side.
            - The shared output-class property should be encoded once in a base IPQ
            inside `getIncrementalInputQueries()` before cloning per-point IPQs.
            - Division-and-Conquer (DnC) and parallelism are intentionally disabled.
        """
        # --- 0) Sanity checks
        if not getattr(self, "incremental_mode", False):
            raise RuntimeError(
                "incrementalSolve called but incremental_mode is False. "
                "Call addRobustnessBatch(points, epsilon) first."
            )

        if propertyFilename:
            raise NotImplementedError(
                "propertyFilename is not supported in incrementalSolve() yet. "
                "Pass an empty string '' for now."
            )

        if options is None:
            options = MarabouCore.Options()

        # Require incremental flag to exist and be settable.
        options._incremental = True

        # DnC must be off: use _snc (the pybind name for DNC_MODE)
        if hasattr(options, "_snc") and options._snc:
            raise ValueError("Incremental mode does not support DnC/parallelism. Please set options._snc = False.")
        assert not getattr(options, "_snc", False), "Incremental mode does not support DnC/parallelism"

        # --- 1) Build dependency analyzer:
        base_ipq = self.getInputQuery()

        # flatten input vars to set bounds in the same order as getInputQuery()
        flat_inputs = []
        for inputVarArray in self.inputVars:
            for inputVar in inputVarArray.flatten():
                flat_inputs.append(int(inputVar))

        cover_lb, cover_ub = self.getCoveringInputBounds()
        # set covering bounds on base_ipq
        for i, v in enumerate(flat_inputs):
            base_ipq.setLowerBound(v, cover_lb[i])
            base_ipq.setUpperBound(v, cover_ub[i])

        # analyzer = MarabouCore.buildDependencyAnalyzer(base_ipq)
        all_lbs = self.incremental_input_lbs.tolist()   # shape (Q, n)
        all_ubs = self.incremental_input_ubs.tolist()   # shape (Q, n)
        analyzer = MarabouCore.buildDependencyAnalyzer(base_ipq, all_lbs, all_ubs)

        # --- 2) Build per-point IPQs (shared outputs; per-point input bounds)
        ipqs = self.getIncrementalInputQueries()
        if not isinstance(ipqs, (list, tuple)) or len(ipqs) == 0:
            raise RuntimeError(
                "getIncrementalInputQueries() returned no IPQs. "
                "Ensure addRobustnessBatch() was called and the builder is implemented."
            )

        # --- 3) Solve each point sequentially (analyzer lifetime spans the loop)
        exitCodes, valsList, statsList = [], [], []

        for idx, ipq in enumerate(ipqs):
            per_point_filename = ""
            if filename:
                per_point_filename = f"{filename}.pt{idx:03d}"

            ipq.setDependencyAnalyzer(analyzer)
            exitCode, vals, stats = MarabouCore.solve(ipq, options, str(per_point_filename))
            exitCodes.append(exitCode)
            valsList.append(vals)
            statsList.append(stats)

            if verbose:
                print(f"[{idx}] {exitCode}")
                if exitCode == "sat":
                    # Mirror solve(): print inputs then outputs
                    for j in range(len(self.inputVars)):
                        for i in range(self.inputVars[j].size):
                            v = self.inputVars[j].item(i)
                            if v in vals:
                                print(f"  input {i} = {vals[v]}")
                    for j in range(len(self.outputVars)):
                        for i in range(self.outputVars[j].size):
                            v = self.outputVars[j].item(i)
                            if v in vals:
                                print(f"  output {i} = {vals[v]}")

        return [exitCodes, valsList, statsList]


    def calculateBounds(self, filename="", verbose=True, options=None):
        """Function to calculate bounds represented by this network

        Args:
            filename (string): Path for redirecting output
            verbose (bool): If true, print out output bounds after calculation finishes
            options (:class:`~maraboupy.MarabouCore.Options`): Object for specifying Marabou options, defaults to None

        Returns:
            (tuple): tuple containing:
                - exitCode (str): A string representing the exit code. Only unsat can be return.
                - bounds (Dict[int, tuple]): Empty dictionary if UNSAT, otherwise a dictionary of bounds for output variables
                - stats (:class:`~maraboupy.MarabouCore.Statistics`): A Statistics object to how Marabou performed
        """
        ipq = self.getInputQuery()
        if options == None:
            options = MarabouCore.Options()
        exitCode, bounds, stats = MarabouCore.calculateBounds(ipq, options, str(filename))

        if verbose:
            print(exitCode)
            if exitCode == "":
                for j in range(len(self.outputVars)):
                    for i in range(self.outputVars[j].size):
                        print("output bounds {} = {}".format(i, bounds[self.outputVars[j].item(i)]))

        return [exitCode, bounds, stats]


    def evaluateWithMarabou(self, inputValues, filename="evaluateWithMarabou.log", options=None):
        """Function to evaluate network at a given point using Marabou as solver

        Args:
            inputValues (list of np arrays): Inputs to evaluate
            filename (str): Path to redirect output if using Marabou solver, defaults to "evaluateWithMarabou.log"
            options (:class:`~maraboupy.MarabouCore.Options`): Object for specifying Marabou options, defaults to None

        Returns:
            (list of np arrays): Values representing the outputs of the network or None if system is UNSAT
        """
        # Make sure inputValues is a list of np arrays and not list of lists
        inputValues = [np.array(inVal) for inVal in inputValues]
        
        inputVars = self.inputVars # list of numpy arrays
        outputVars = self.outputVars # list of numpy arrays

        inputDict = dict()
        inputVarList = np.concatenate([inVar.flatten() for inVar in inputVars], axis=-1).flatten()
        inputValList = np.concatenate([inVal.flatten() for inVal in inputValues]).flatten()
        assignList = zip(inputVarList, inputValList)
        for x in assignList:
            inputDict[x[0]] = x[1]

        ipq = self.getInputQuery()
        for k in inputDict:
            ipq.setLowerBound(k, inputDict[k])
            ipq.setUpperBound(k, inputDict[k])

        if options == None:
            options = MarabouCore.Options()
        exitCode, outputDict, _ = MarabouCore.solve(ipq, options, str(filename))

        # When the query is UNSAT an empty dictionary is returned
        if outputDict == {}:
            return None

        outputValues = [outVars.reshape(-1).astype(np.float64) for outVars in outputVars]
        for i in range(len(outputValues)):
            for j in range(len(outputValues[i])):
                outputValues[i][j] = outputDict[outputValues[i][j]]
            outputValues[i] = outputValues[i].reshape(outputVars[i].shape)
        return outputValues

    def evaluate(self, inputValues, useMarabou=True, options=None, filename="evaluateWithMarabou.log"):
        """Function to evaluate network at a given point

        Args:
            inputValues (list of np arrays): Inputs to evaluate
            useMarabou (bool): Whether to use Marabou solver or TF/ONNX, defaults to True
            options (:class:`~maraboupy.MarabouCore.Options`): Object for specifying Marabou options, defaults to None
            filename (str): Path to redirect output if using Marabou solver, defaults to "evaluateWithMarabou.log"

        Returns:
            (list of np arrays): Values representing the outputs of the network or None if output cannot be computed
        """
        if useMarabou:
            return self.evaluateWithMarabou(inputValues, filename=filename, options=options)
        if not useMarabou:
            return self.evaluateWithoutMarabou(inputValues)

    def findError(self, inputValues, options=None, filename="evaluateWithMarabou.log"):
        """Function to find error between Marabou solver and TF/Nnet at a given point

        Args:
            inputValues (list of np arrays): Input values to evaluate
            options (:class:`~maraboupy.MarabouCore.Options`) Object for specifying Marabou options, defaults to None
            filename (str): Path to redirect output if using Marabou solver, defaults to "evaluateWithMarabou.log"

        Returns:
            (list of np arrays): Values representing the error in each output variable
        """
        outMar = self.evaluate(inputValues, useMarabou=True, options=options, filename=filename)
        outNotMar = self.evaluate(inputValues, useMarabou=False, options=options, filename=filename)
        assert len(outMar) == len(outNotMar)
        err = [np.abs(outMar[i] - outNotMar[i]) for i in range(len(outMar))]
        return err
