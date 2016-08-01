#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>

Camera::Camera( const char* name, Object *parent )
    : Object( name, parent ) {
    // Some sensible defaults

    near =1.f;
    far =100.f;
    fov =50.f;
    aspect =1.f;
    position =glm::vec3( 1.f, .3f, -1.f );
    target =glm::vec3(0);
    up =glm::vec3(0,1,0);
}

Camera::~Camera() {}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt( position, target, up );
}

glm::mat4 Camera::getProjMatrix() const {
    return glm::perspective( glm::radians( fov ), aspect, near, far );

}

Variant 
Camera::callMeta( const std::string& method, const Variant::list& args ) {

}

bool 
Camera::setMeta( const std::string& property, const Variant& value ) {
}

Variant 
Camera::getMeta( const std::string& property ) const {

}
