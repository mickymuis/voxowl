#pragma once

#include "object.h"
#include "glm/mat4x4.hpp"

class Actor : public Object { 
    public:
        Actor( const char* name, Object* parent =0 );
        ~Actor();

        glm::mat4 modelMatrix() { return glm::mat4(1.f); }

        // TODO: fill from Folia
};
