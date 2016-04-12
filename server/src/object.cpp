#include "object.h"
#include "parser.h"

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
    return list;
}


Object *
Object::getChildByName( const std::string& name ) const {
    if( name.empty() )
        return (Object*)this;
    size_t pos = name.find_first_of( MEMBER_CHARS );
    std::string first, rest;
    if( pos == std::string::npos )
        first =name;
    else {
        first =name.substr( 0, pos );
        rest =name.substr( pos+1 );
    }

    for( Objects::iterator it =children.begin(); it != children.end(); it++ ) {
        if( !(*it)->getName().empty() && (*it)->getName() == first )
             return (*it)->getChildByName( rest );
    }
    
    return 0;
}

void 
Object::update( float deltatime ) {
    for( Objects::iterator it =children.begin(); it != children.end(); it++ )
        (*it)->update( deltatime );
}

stringlist_t Object::listMeta( META_TYPE meta, const std::string& name ) const {
    Object *obj =getChildByName( name );
    if( obj )
        return obj->listMeta( meta );
    stringlist_t list;
    return list;
}

bool 
Object::setMeta( const std::string& property, const std::string& value ) {
    return false;
}

std::string 
Object::getMeta( const std::string& property ) const {
    return std::string();
}

stringlist_t 
Object::listMeta( META_TYPE t ) const {
    switch( t ) {
        case META_NONE:
            break;
        case META_CHILD:
            return listChildren();
        case META_PROPERTY:
            return property_list;
        case META_METHOD:
            return method_list;
    }
    return stringlist_t();
}

bool 
Object::callMeta( const std::string&, MetaArgument::list ) {
    return false;
}


ObjectFactory::list ObjectFactory::factory_list;
