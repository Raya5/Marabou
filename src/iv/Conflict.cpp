#include "Conflict.h"

#include <cassert>

Conflict::Conflict( double epsilon,
                    const std::vector<unsigned> &vars,
                    const std::vector<bool> &isActive )
    : _epsilon( epsilon )
    , _vars( vars )
    , _isActive( isActive )
{
    assert( _vars.size() == _isActive.size() );
}

double Conflict::getEpsilon() const
{
    return _epsilon;
}

const std::vector<unsigned> &Conflict::getVars() const
{
    return _vars;
}

const std::vector<bool> &Conflict::getIsActive() const
{
    return _isActive;
}
