#pragma once

#include "../core/application.h"

class ServerModule : public Module {
    public:
        static Descriptor getDescriptor();
        int exec( const arglist_t& );
};
