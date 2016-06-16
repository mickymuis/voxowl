#include "actor.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"

Actor::Actor( const char* name, Object* parent )
    : Object( name, parent ),
    m_position( 0.0, 0.0, 0.0 ),
    m_rotation( 0.0, 0.0, 0.0 ),
    m_scale( 1.0 ),
    m_orbit ( 0.0, 0.0, 0.0 ),
    m_matCache ( 1.0 ),
    m_uptodate ( false ), 
    m_v_lin ( 0.0 ),
    m_v_ang ( 0.0 ),
    m_v_rot ( 0.0 ),
    m_mass ( 1.0 ) {

    addMethod( "setRotation" );
    addMethod( "setPosition" );
    addMethod( "setScale" );
    addMethod( "setOrbit" );
    addMethod( "translate" );
    addMethod( "rotate" );
    addMethod( "rotateAround" );
}

Actor::~Actor() {}

glm::mat4 
Actor::modelMatrix() {

    if( !m_uptodate ) {
        glm::mat4 mat( 1.0 );
                    
            mat =glm::scale( mat, glm::vec3( m_scale ) );
            mat =glm::rotate( mat, 
                    m_orbit.x, glm::vec3( 1.0, 0.0, 0.0 ) );
            mat =glm::rotate( mat, 
                     m_orbit.y, glm::vec3( 0.0, 1.0, 0.0 ) );
            mat =glm::rotate( mat, 
                    m_orbit.z, glm::vec3( 0.0, 0.0, 1.0 ) );
            
            mat =glm::translate( mat, m_position );
            mat =glm::rotate( mat, 
                    m_rotation.x, glm::vec3( 1.0, 0.0, 0.0 ) );
            mat =glm::rotate( mat, 
                    m_rotation.y, glm::vec3( 0.0, 1.0, 0.0 ) );
            mat =glm::rotate( mat, 
                    m_rotation.z, glm::vec3( 0.0, 0.0, 1.0 ) );
                    
            m_matCache =mat;

            m_uptodate =true;
    }
    return m_matCache;
}

void 
Actor::update( float deltatime ) {

    translate( deltatime * m_v_lin );
    rotate( deltatime * m_v_rot );
    rotateAround( deltatime * m_v_ang );
}

Variant 
Actor::callMeta( const std::string& method, const Variant::list& args ) {
    if( method == "setPosition" ) {
        setPosition( Variant::toVec3( args )) ;
    } else if( method == "setRotation" ) {
        setRotation( Variant::toVec3( args )) ;
    } else if( method == "setScale" ) {
        float f =0.f;
        if( args.size() > 0 )
            f =(float)args[0].toReal();
        setScale( f ) ;
    } else if( method == "setOrbit" ) {
        setOrbit( Variant::toVec3( args )) ;
    } else if( method == "translate" ) {
        translate( Variant::toVec3( args )) ;
    } else if( method == "rotate" ) {
        rotate( Variant::toVec3( args )) ;
    } else if( method == "rotateAround" ) {
        rotateAround( Variant::toVec3( args )) ;
    } else
        return Object::callMeta( method, args );

    return Variant();
}

