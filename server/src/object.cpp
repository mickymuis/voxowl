#include "object.h"
#include "parser.h"

std::string 
Variant::toString() {
    switch( _type ) {
        case TYPE_REAL:
            return std::to_string( (long double)value_real );
            break;
        case TYPE_STRING:
            return value_string;
            break;
        case TYPE_OBJECT:
            if( value_ptr ) 
                return value_ptr->getName();
            break;
        default:
            break;
    }
    return std::string();
}

double 
Variant::toReal() {
    switch( _type ) {
        case TYPE_REAL:
            return value_real;
            break;
        case TYPE_STRING:
            return atof( value_string.c_str() );
            break;
        case TYPE_OBJECT:
            break;
        default:
            break;
    }
    return 0.;
}

Object* 
Variant::toObject() {
    switch( _type ) {
        case TYPE_REAL:
            break;
        case TYPE_STRING:
            break;
        case TYPE_OBJECT:
            return value_ptr;
            break;
        default:
            break;
    }
    return 0;
}
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

stringlist_t Object::listMeta( META_TYPE meta, const std::string& reference ) const {
    Object *obj =getChildByName( reference );
    if( obj )
        return obj->listMeta( meta );
    stringlist_t list;
    return list;
}

bool 
Object::setMeta( const std::string& property, const Variant& value ) {
    return false;
}

Variant 
Object::getMeta( const std::string& property ) const {
    return Variant();
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
Object::hasMeta( META_TYPE type, const std::string& name ) const {
    stringlist_t list = listMeta( type );
    return std::find( list.begin(), list.end(), name ) != list.end();
}

bool 
Object::hasMeta( META_TYPE type, const std::string& reference, const std::string& name ) const {
    stringlist_t list = listMeta( type, reference );
    return std::find( list.begin(), list.end(), name ) != list.end();
}

bool 
Object::callMeta( const std::string&, Variant::list ) {
    return false;
}


ObjectFactory::list ObjectFactory::factory_list;
