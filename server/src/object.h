#pragma once

#include <string>
#include <set>
#include <stdint.h>
#include "types.h"

class Object {
    public:
        enum META_TYPE {
            META_NONE,
            META_PROPERTY,
            META_METHOD,
            META_CHILD
        };
        Object( const std::string& name, Object* parent =0 );
        ~Object();
    
        Object* getParent( ) const;
        std::string getName( ) const;
        virtual void update( float deltatime );
        
        typedef std::set<Object*> Objects;
        Objects getChildren( ) const;
        stringlist_t listChildren( ) const;

        virtual bool setMeta( const std::string& property, const std::string& value ) =0;
        virtual std::string getMeta( const std::string& property ) const =0;
        virtual stringlist_t listMeta( META_TYPE ) const =0;
        virtual stringlist_t listMeta( META_TYPE, const std::string& name ) const;
        virtual bool callMeta( const std::string& method ) =0;
        
        /*virtual int RTTI() const =0;
        inline bool isA( int type ) const { return type == RTTI(); }*/
    
    protected:
        Objects children;
        Object* parent;
        std::string name;
    
};

class ObjectFactory {
    public:
        ObjectFactory( const std::string& name ) : name( name ) {}

        virtual Object* create( const std::string& name, Object* parent ) =0;

        std::string getName() const { return name; }

    protected:
        std::string name;
};

template<class OBJ>
class ObjectFactoryT : public ObjectFactory {
    public:
        ObjectFactoryT( const std::string& name ) : ObjectFactory( name ) {}

        Object* create( const std::string& name, Object* parent ) { return new OBJ( name, parent ); }
};

