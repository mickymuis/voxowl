#pragma once

#include "actor.h"
#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"

class Camera : public Actor {
    public:
        Camera( const char* name, Object *parent =0 );
        ~Camera();

        void setPosition( const glm::vec3& v ) { position =v; }
        glm::vec3 getPosition() const { return position; }

        void setViewTarget( const glm::vec3& v ) { target =v; } 
        glm::vec3 getViewTarget() const { return target; }

        void setFov( float f ) { fov =f; }
        float getFov() const { return fov; }

        void setNearBound( float f ) { near =f; }
        float getNearBound( ) const { return near; }

        void setFarBound( float f ) { far =f; }
        float getFarBound() const { return far; }

        void setAspect( float f ) { aspect =f; }
        float getAspect() const { return aspect; } 

        void setUp( const glm::vec3 &v ) { up =v; }
        glm::vec3 getUp() { return up; }
        
        glm::vec3 getViewDirection() const { return target-position; }

        glm::mat4 getViewMatrix();
        glm::mat4 getProjMatrix() const;

        virtual Variant callMeta( const std::string& method, const Variant::list& args );
        virtual bool setMeta( const std::string& property, const Variant& value );
        virtual Variant getMeta( const std::string& property ) const;

    private:
        glm::vec3 position, target, up;
        float fov, near, far, aspect;
};
