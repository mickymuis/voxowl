#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>
#include <stdio.h>

Camera::Camera( const char* name, Object *parent )
    : Actor( name, parent ) {
    // Some sensible defaults

    near =1.f;
    far =100.f;
    fov =50.f;
    aspect =1.f;
    position =glm::vec3( 1.f, .3f, -1.f );
    translate( glm::vec3( .01f,.01f,-1.f ) );
    target =glm::vec3(0);
    up =glm::vec3(0,1,0);
}

Camera::~Camera() {}

glm::mat4 Camera::getViewMatrix() {
    glm::vec3 p =glm::vec3( modelMatrix() * glm::vec4(0,0,0,1) );
    printf( "Camera position (%f,%f,%f)\n", p.x, p.y, p.z );
    return glm::lookAt( p, target, up );
}

glm::mat4 Camera::getProjMatrix() const {
    return glm::perspective( glm::radians( fov ), aspect, near, far );

}

Variant 
Camera::callMeta( const std::string& method, const Variant::list& args ) {
    return Actor::callMeta( method, args );
}

bool 
Camera::setMeta( const std::string& property, const Variant& value ) {
    return Actor::setMeta( property, value );
}

Variant 
Camera::getMeta( const std::string& property ) const {
    return Actor::getMeta( property );
}
