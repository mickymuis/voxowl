#include "object.h"

Object::Object( const std::string& name, Object* parent ) : parent( parent ), name( name ) {
    if( parent )
        parent->children.insert( this );
}

Object::~Object() {
    Objects cpy =children;
    for( Objects::iterator it =cpy.begin(); it != cpy.end(); it++ )
        delete (*it);
            
    if( parent )
        parent->children.erase( this );
}
    
Object*
Object::getParent( ) const { return parent; }

std::string
Object::getName( ) const { return name; }

Object::Objects
Object::getChildren( ) const {
    return children;
}

stringlist_t Object::listChildren( ) const {
    stringlist_t list;
    for( Objects::iterator it =children.begin(); it != children.end(); it++ ) {
        if( !(*it)->getName().empty() )
            list.push_back( (*it)->getName() );
    }
}

void 
Object::update( float deltatime ) {
    for( Objects::iterator it =children.begin(); it != children.end(); it++ )
        (*it)->update( deltatime );
}

stringlist_t Object::listMeta( META_TYPE meta, const std::string& name ) const {
    size_t pos =name.find_first_of( ".:/\\" );
    std::string pname = name.substr( 0, pos-1 );
    std::string qname = name.substr( pos+1 );
    for( Objects::iterator it =children.begin(); it != children.end(); it++ ) {
        if( !(*it)->getName().empty() && (*it)->getName() == pname )
             return (*it)->listMeta( meta, qname );
    }
    stringlist_t list;
    return list;
}

