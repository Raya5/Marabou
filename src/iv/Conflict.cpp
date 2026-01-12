#include "Conflict.h"

#include <cassert>

Conflict::Conflict( const std::vector<unsigned> &vars,
                    const std::vector<bool> &isActive )
    : _vars( vars )
    , _isActive( isActive )
{
    assert( _vars.size() == _isActive.size() );
}

const std::vector<unsigned> &Conflict::getVars() const
{
    return _vars;
}

const std::vector<bool> &Conflict::getIsActive() const
{
    return _isActive;
}
