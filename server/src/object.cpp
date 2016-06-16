#include "object.h"
#include "parser.h"

std::string 
Variant::toString() const{
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
Variant::toReal() const {
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

bool 
Variant::toBool() const {
    switch( _type ) {
        case TYPE_REAL:
            return (bool)value_real;
            break;
        case TYPE_STRING:
            return (bool)atoi( value_string.c_str() );
            break;
        case TYPE_OBJECT:
            if( value_ptr )
                return true;
            break;
        default:
            break;
    }
    return false;
}

int
Variant::toInt() const {
    switch( _type ) {
        case TYPE_REAL:
            return (int)value_real;
            break;
        case TYPE_STRING:
            return atoi( value_string.c_str() );
            break;
        case TYPE_OBJECT:
            break;
        default:
            break;
    }
    return 0;
}

Object* 
Variant::toObject() const {
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
glm::vec2 
Variant::toVec2( const Variant::list& vl ) {
    glm::vec2 v(0);
    if( vl.size() > 0 )
        v.x = (float)vl[0].toReal();
    if( vl.size() > 1 )
        v.y = (float)vl[1].toReal();
    return v;
}

glm::vec3 
Variant::toVec3( const Variant::list& vl ) {
    glm::vec3 v( toVec2( vl ), 0 );
    if( vl.size() > 2 )
        v.z =(float)vl[2].toReal();
    return v;
}

glm::vec4 
Variant::toVec4( const Variant::list& vl ) {
    glm::vec4 v( toVec3( vl ), 0 );
    if( vl.size() > 3 )
        v.w =(float)vl[3].toReal();
    return v;
}

std::ostream & operator<<(std::ostream &os, const Variant& v)
{
        switch( v.type() ) {
            case Variant::TYPE_REAL:
                return os << v.value_real;
            case Variant::TYPE_STRING:
                return os << v.value_string;
            case Variant::TYPE_OBJECT:
                if( v.value_ptr )
                    return os << "<" << v.value_ptr->getName() << ">";
                else
                    return os << "<undefined>";
            default:
                break;
        }
        return os;
}

// Implementation of class Object

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

stringlist_t 
Object::listMeta( META_TYPE meta, const std::string& reference ) const {
    Object *obj =getChildByName( reference );
    if( obj )
        return obj->listMeta( meta );
    stringlist_t list;
    return list;
}

bool 
Object::setMeta( const std::string& property, const Variant& value ) {
   if( !hasProperty( property ) )
       return false;
   property_list[property] =value;
   return true;
}

Variant 
Object::getMeta( const std::string& property ) const {
    if( !hasProperty( property ) )
        return Variant();
    return property_list.at(property);
}

stringlist_t 
Object::listMeta( META_TYPE t ) const {
    switch( t ) {
        case META_CHILD:
            return listChildren();
        case META_PROPERTY:
            return listProperties();
        case META_METHOD:
            return method_list;
        default: break;
    }
    return stringlist_t();
}

bool 
Object::hasMeta( META_TYPE type, const std::string& name ) const {
    if( type == META_PROPERTY )
        return hasProperty( name );
    stringlist_t list = listMeta( type );
    return std::find( list.begin(), list.end(), name ) != list.end();
}

bool 
Object::hasMeta( META_TYPE type, const std::string& reference, const std::string& name ) const {
    if( type == META_PROPERTY ) {    
        Object *obj =getChildByName( reference );
        if( obj )
            return obj->hasProperty( name );
        return false;
    }
    stringlist_t list = listMeta( type, reference );
    return std::find( list.begin(), list.end(), name ) != list.end();
}

Variant 
Object::callMeta( const std::string&, const Variant::list& ) {
    return Variant();
}

bool
Object::hasProperty( const std::string& name ) const {
    return property_list.count( name ) != 0;
}

void
Object::addProperty( const std::string& name, Variant v ) {
    if( !hasProperty( name ) )
        property_list[name] = v;
}

void
Object::removeProperty( const std::string& name ) {
    if( hasProperty( name ) )
        property_list.erase( name );
}

stringlist_t
Object::listProperties() const {
    PropertyList::const_iterator it;
    stringlist_t list;
    for( it = property_list.begin(); it != property_list.end(); it++ )
        list.push_back( it->first );
    return list;
}

void
Object::addMethod( const std::string& name ) {
    if( !hasMeta( META_METHOD, name ) ) {
        method_list.push_back( name );
    }
}

ObjectFactory::list ObjectFactory::factory_list;
