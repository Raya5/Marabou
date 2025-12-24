#ifndef __Conflict_h__
#define __Conflict_h__

#include <vector>

/*
 * Conflict
 * --------
 * At design level, a Conflict is one learned reason that rules out a
 * combination of ReLU phases / decisions that the solver discovered
 * is impossible under the current query.
 *
 * Conceptually, this is a SAT-style learned clause.
 *
 * Each conflict:
 *   - Is learned under a specific epsilon
 *   - Consists of a list of phase variables (ints)
 *   - For each variable, records whether the phase is active or inactive
 *
 * Representation:
 *   - _vars[i]       : variable index (ReLU / phase variable)
 *   - _isActive[i]   : true  -> active phase
 *                      false -> inactive phase
 *
 * This class is immutable after construction.
 * No reasoning or propagation happens here.
 */

class Conflict
{
public:
    /*
     * Construct a conflict learned at epsilon, with a fixed list of literals.
     *
     * Preconditions:
     *   - vars.size() == isActive.size()
     */
    Conflict( double epsilon,
              const std::vector<unsigned> &vars,
              const std::vector<bool> &isActive );

    /*
     * Accessors
     */
    double getEpsilon() const;
    const std::vector<unsigned> &getVars() const;
    const std::vector<bool> &getIsActive() const;

private:
    double _epsilon;                  // epsilon at which conflict was learned
    std::vector<unsigned> _vars;      // phase variable indices
    std::vector<bool> _isActive;      // corresponding phase values
};

#endif // __Conflict_h__
