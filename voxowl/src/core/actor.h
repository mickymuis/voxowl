#pragma once

#include "object.h"
#include "glm/mat4x4.hpp"
#include "glm/vec3.hpp"

class Actor : public Object { 
    public:
        Actor( const char* name, Object* parent =0 );
        ~Actor();
        
        /* Absolute functions */
        void setPosition( glm::vec3 p ) { m_position = p;  m_uptodate =false; }
        glm::vec3 position() const { return m_position; }
        
        void setRotation( glm::vec3 r ) { m_rotation = r; m_uptodate =false; }
        glm::vec3 rotation() const { return m_rotation; }
        
        void setScale( float s ) { m_scale = s; m_uptodate =false; }
        float scale() const { return m_scale; };
        
        void setOrbit( glm::vec3 o ) { m_orbit = o; m_uptodate =false; }
        glm::vec3 orbit() const { return m_orbit; }
        
        /* Relative utility functions */
        void translate( glm::vec3 t ) { m_position += t; m_uptodate =false; }
        void rotate( glm::vec3 r ) { m_rotation += r; m_uptodate =false; }
        void rotateAround( glm::vec3 r ) { m_orbit += r; m_uptodate =false; }
        
        /* Matrix generation */
        glm::mat4 modelMatrix();

        /* Continuous functions */
        void setLinearVelocity( glm::vec3 v ) { m_v_lin = v; }
        glm::vec3 linearVelocity() { return m_v_lin; }
        
        void setAngularVelocity( glm::vec3 v ) { m_v_ang = v; }
        glm::vec3 angularVelocity() { return m_v_ang; }
        
        void setRotationalVelocity( glm::vec3 v ) { m_v_rot = v; }
        glm::vec3 rotationalVelocity() { return m_v_rot; }
        
        void setMass( float m ) { m_mass =m; }
        float mass() { return m_mass; }
        
        void update( float deltatime );
        virtual Variant callMeta( const std::string& method, const Variant::list& args );
        
private:
        
        glm::vec3 m_position;
        glm::vec3 m_rotation;
        float m_scale;
        glm::vec3 m_orbit;
        glm::mat4 m_matCache;
        bool m_uptodate;

        glm::vec3 m_v_lin;
        glm::vec3 m_v_ang;
        glm::vec3 m_v_rot;
        float m_mass;
};
