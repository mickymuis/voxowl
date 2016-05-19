#pragma once

#include <string>
#include <set>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>
#include "types.h"

class Object;

class Variant {
    public:
        typedef std::vector<Variant> list;
        enum TYPE {
            TYPE_NONE,
            TYPE_REAL,
            TYPE_STRING,
            TYPE_OBJECT
        };

        Variant() : _type( TYPE_NONE ) {}
        Variant( TYPE t ) : _type( t ) {}
        Variant( const std::string & str ) : _type( TYPE_STRING ), value_string( str ) {}
        Variant( double d ) : _type( TYPE_REAL ), value_real( d ) {}
        Variant( unsigned int i ) : _type( TYPE_REAL ), value_real( (double)i ) {}
        Variant( int i ) : _type( TYPE_REAL ), value_real( (double)i ) {}
        Variant( Object* obj ) : _type( TYPE_OBJECT ), value_ptr( obj ) {}

        TYPE type() const { return _type; }
        bool isA( TYPE t ) const { return _type == t; }

        void set( const std::string& str ) { _type = TYPE_STRING; value_string = str; }
        void set( double d ) { _type = TYPE_REAL; value_real = d; }
        void set( Object* obj ) { _type = TYPE_OBJECT; value_ptr= obj; }

        std::string toString() const;
        double toReal() const;
        bool toBool() const;
        int toInt() const;
        Object* toObject() const;

    private:
        TYPE _type;
        double value_real;
        std::string value_string;
        Object *value_ptr;

        friend std::ostream & operator<<(std::ostream &os, const Variant& v);
};

std::ostream & operator<<(std::ostream &os, const Variant& v);

class Object {
    public:
        enum META_TYPE {
            META_NONE =0x0,
            META_PROPERTY =0x1,
            META_METHOD =0x2,
            META_CHILD =0x4,
            META_ALL =0xf
        };
        Object( const std::string& name, Object* parent =0 );
        ~Object();
    
        Object* getParent( ) const;
        std::string getName( ) const;
        virtual void update( float deltatime );
        
        typedef std::set<Object*> Objects;
        typedef std::map<std::string, Variant> PropertyList;
        Objects getChildren( ) const;
        Object* getChildByName( const std::string& name ) const;
        stringlist_t listChildren( ) const;

        virtual bool setMeta( const std::string& property, const Variant& value );
        virtual Variant getMeta( const std::string& property ) const;
        stringlist_t listMeta( META_TYPE ) const;
        stringlist_t listMeta( META_TYPE, const std::string& name ) const;
        bool hasMeta( META_TYPE, const std::string& reference ) const;
        bool hasMeta( META_TYPE, const std::string& reference, const std::string& name ) const;
        virtual Variant callMeta( const std::string& method, const Variant::list& args );
        
    
    protected:
        Object( const Object& rhs ) {}
        void operator=( const Object& rhs ) {}
        virtual bool hasProperty( const std::string& ) const;
        virtual void addProperty( const std::string&, Variant = Variant() );
        virtual void removeProperty( const std::string& );
        virtual stringlist_t listProperties() const;
        virtual void addMethod( const std::string& );

        Objects children;
        Object* parent;
        std::string name;
        PropertyList property_list;
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

