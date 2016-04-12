#pragma once

#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include "types.h"

class MetaArgument {
    public:
        typedef std::vector<MetaArgument> list;
        enum ARGTYPE {
            ARGTYPE_REAL,
            ARGTYPE_STRING
        };

        ARGTYPE type;
        double value_real;
        std::string value_string;
};

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
        Object* getChildByName( const std::string& name ) const;
        stringlist_t listChildren( ) const;

        virtual bool setMeta( const std::string& property, const std::string& value );
        virtual std::string getMeta( const std::string& property ) const;
        stringlist_t listMeta( META_TYPE ) const;
        stringlist_t listMeta( META_TYPE, const std::string& name ) const;
        virtual bool callMeta( const std::string& method, MetaArgument::list args );
        
        /*virtual int RTTI() const =0;
        inline bool isA( int type ) const { return type == RTTI(); }*/
    
    protected:
        Objects children;
        Object* parent;
        std::string name;
        stringlist_t property_list;
        stringlist_t method_list;
    
};

class ObjectFactory {
    public:
        typedef std::vector<ObjectFactory*> list;
        ObjectFactory( const std::string& name ) : name( name ) {}

        virtual Object* create( const std::string& name, Object* parent ) =0;

        std::string getName() const { return name; }

        static ObjectFactory* getFactory( int i ) { return factory_list[i]; }
        static int getFactoryCount() { return factory_list.size(); }
        static void newFactory( ObjectFactory *fac ) { factory_list.push_back( fac ); }
        static void deleteFactory( ObjectFactory *fac ) { 
            factory_list.erase( std::find( factory_list.begin(), factory_list.end(), fac ) ); 
            delete fac; 
        }
        static ObjectFactory* getFactoryByName( const std::string& name ) {
            for( list::iterator fac = factory_list.begin(); fac != factory_list.end(); fac++ )
                if( (*fac)->getName() == name )
                    return (*fac);
            return 0;
        }

    protected:
        std::string name;

    private:
        static list factory_list;
};

template<class OBJ>
class ObjectFactoryT : public ObjectFactory {
    public:
        ObjectFactoryT( const std::string& name ) : ObjectFactory( name ) {}

        Object* create( const std::string& name, Object* parent ) { return new OBJ( name, parent ); }
};

