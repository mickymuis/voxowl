#include <iostream>
#include "core/voxowl.h"
#include "core/application.h"
#include "util/performance_counter.h"
#include "module/server_module.h"

void 
printHelp( const Application &a ) {
    std::cerr << VOXOWL_VERSION_FULL_NAME << std::endl
        << "Usage: `" << a.getCmdArg() << " " 
        << (a.getModuleArg().empty() ? "<module_name>" : a.getModuleArg())
        << " arguments ...'" << std::endl;

    if( a.getModuleArg().empty() ) {
        std::cerr << "List of available modules: " << std::endl;
        a.printModuleList( std::cerr );
        std::cerr << "Type `" << a.getCmdArg() << " <module_name> --help' for more information" << std::endl;
    } else {
        std::cerr << "Help for module `" << a.getModuleArg() << "':" << std::endl
            << a.getDescriptor( a.getModuleArg() ).helpText << std::endl;
    }

/*    arglist_t args =a.getParsedArgs();
    for( const auto& pair : args ) {
        std::cerr << "\t" << pair.first << "\t\t" << pair.second << std::endl;
    }*/
}

void 
setupEnvironment( Object* env, const Application& a ) {


}

int 
main( int argc, const char** argv ) {
    Object env( "root", nullptr );
    Application a( &env );
    a.registerModule( ServerModule::getDescriptor(), new DerivedCreator<Module,ServerModule>() );
    
    bool b =a.parseArgs( argc, argv );

    if( !a.getModuleArg().empty() && !a.isRegistered( a.getModuleArg() ) ) {
        std::cerr << "Module `" << a.getModuleArg() << "' is not available. Perhaps " 
            << VOXOWL_VERSION_NAME << " was compiled without it?" << std::endl;
        return -1;
    }

    if( !b || a.getParsedArgs().count( "help" ) > 0 ) {
        printHelp( a );
        return 0;
    }

    int ret = a.exec( a.getModuleArg() );

    PerformanceCounter::printAll( std::cout );
    PerformanceCounter::cleanup();
    
    return ret;
}
