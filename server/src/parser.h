#pragma once

#include <streambuf>
#include <iostream>
#include "object.h"

#define MAX_COMMAND_LEN 1024

class Parser {
    public:
        Parser( std::basic_streambuf<char>* buffer, Object * =0 );
        ~Parser();

        void setEnvironment( Object* env ) { environment =env; }

        bool evaluateNext();
        bool evaluate( stringlist_t );

    private:
        std::basic_streambuf<char>* iobuf;
        Object *environment;
        std::iostream stream;

        
        static stringlist_t tokenize( char * );
        static size_t tokencpy( char *dst, const char* src );
        static size_t tokenlen( const char * );
        static char *tokennext( char** );
};
