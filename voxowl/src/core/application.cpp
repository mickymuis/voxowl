#include "application.h"
#include <algorithm>
#include <string.h>

#define MULTIC_ARG_LEADER "--"
#define SINGLC_ARG_LEADER "-"
#define UNNAMED_ARG "#"
#define ARG_ASSIGN '='

Variant::list 
unnamedArgs( const arglist_t& args ) {
    Variant::list list;
    int i =0;

    while( 1 ) {
        std::string key ="#" + std::to_string(i);
        if( args.find( key ) == args.end() )
            break;
        list.push_back( args.at(key) );
    }
    return list;
}

Application::Application( Object* environment ) 
    : env( environment ) {}
Application::~Application() {}

static bool 
isMultiCharArg( const char* str ) {
    return strncmp( str, MULTIC_ARG_LEADER, strlen(MULTIC_ARG_LEADER ) ) == 0 ;
}

static bool 
isSingleCharArg( const char* str ) {
    return strncmp( str, SINGLC_ARG_LEADER, strlen(SINGLC_ARG_LEADER ) ) == 0 ;
}

static bool
isAssignment( std::string& arg, std::string& rvalue ) {
    size_t assign =arg.find_first_of( ARG_ASSIGN );
    if( assign != std::string::npos ) {
        rvalue = arg.substr( assign + 1 );
        arg.resize( assign );
        return true;
    }
    return false;
}

static std::string
strToLower( const std::string& str ) {
    std::string lower;
    lower.resize( str.size() );

    std::transform( str.begin(), str.end(), lower.begin(), ::tolower );
    return lower;
}

bool 
Application::parseArgs( int argc, const char** argv ) {
    // Currently, we parse the arguments in a Unix-like fashion
    args.clear();
    argCmd =std::string( argv[0] );
    argModule.clear();

    int unnamed =0;

    enum { SINGLE, MULTI, UNNAMED } type;
    const char *leader;

    for( int i =1; i < argc; i++ ) {
        // Multi-char argument
        if( isMultiCharArg( argv[i] ) ) {
            type =MULTI;
            leader =MULTIC_ARG_LEADER;
        }
        // Single-char argument
        else if( isSingleCharArg( argv[i] ) ) {
            type =SINGLE;
            leader =SINGLC_ARG_LEADER;
        } 
        // Unnamed argument
        else { 
            // Special case: the first argument may give the module name
            if( i == 1 ) {
                argModule =std::string( argv[i] );
            } else {
                args[std::string(UNNAMED_ARG)+std::to_string(unnamed++)] = Variant( argv[i] );

            }
            continue;
        }
        
        std::string arg =argv[i] + strlen( leader );
        std::string rvalue;
        if( !isAssignment( arg, rvalue ) ) {
            if( i+1 < argc && !isSingleCharArg( argv[i+1] ) && !isMultiCharArg( argv[i+1] ) ) {
                rvalue =argv[++i];
            }
        }
        Variant var;
        if( rvalue.empty() )
            var.set( 1 );
        else
            var.set( rvalue );
        

        if( type == MULTI ) {
            args[strToLower(arg)] = var; 
        } else if ( type == SINGLE ) {
            for( int j=0; j < arg.size(); j++ ) {
                args[std::string( 1, arg[j] )] = var;
            }
        }
    }

    if( argModule.empty() ) {
        if( args.find( "module" ) == args.end() )
            return false;
        argModule =args["module"].toString();
        args.erase( "module" );
    }

    return true;
}

void 
Application::registerModule( const Module::Descriptor& d, Creator<Module> *c ) {
    std::string key =strToLower( d.key );

    factory.registerCreator( key, c );
    descrMap[key] =d;
}

void 
Application::printModuleList( std::ostream& str ) const {
    for( const auto& pair : descrMap ) {
        str << "\t[" << pair.second.key << "]\t\t" << pair.second.description << std::endl;
    }
}

bool 
Application::isRegistered( const std::string& moduleName ) {
    const auto it =factory.getMap().find( moduleName );
    return it != factory.getMap().end();
}

const Module::Descriptor& 
Application::getDescriptor( const std::string& moduleName ) const {
    return descrMap.at(moduleName); 
}

int 
Application::exec( const std::string& moduleName ) {
    Module* M =factory.create( moduleName );
    if( M == nullptr ) return -1;

    M->environment =env;
    int ret = M->exec( args );
    delete M;

    return ret;
}
