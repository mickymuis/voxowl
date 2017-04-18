#pragma once

#include "../util/factory.h"
#include "object.h"
#include <string>
#include <map>
#include <iostream>

typedef std::map<std::string,Variant> arglist_t;
Variant::list unnamedArgs( const arglist_t& );

class Application;

class Module {
    public:
        struct Descriptor {
            public:
                std::string key;
                std::string description;
                std::string helpText;
        };
        virtual int exec( const arglist_t &) =0;

    protected:
        friend class Application;
        Object* environment;
};

class Application {
    public:
        Application( Object* environment );
        ~Application();

        bool parseArgs( int argc, const char** argv );
        arglist_t getParsedArgs() const { return args; }
        std::string getModuleArg() const { return argModule; }
        std::string getCmdArg() const { return argCmd; }

        void registerModule( const Module::Descriptor&, Creator<Module> * );
        void printModuleList( std::ostream& ) const;
        bool isRegistered( const std::string& moduleName );

        const Module::Descriptor& getDescriptor( const std::string& moduleName ) const;

        int exec( const std::string& moduleName );

    private:
        Object *env;
        Factory<Module,std::string> factory;
        arglist_t args;
        std::string argModule;
        std::string argCmd;
        std::map<std::string,Module::Descriptor> descrMap;

};
