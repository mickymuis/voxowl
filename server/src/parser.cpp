#include "parser.h"
#include <string.h>

Parser::Parser( std::basic_streambuf<char> *buf, Object* env )
    : iobuf( buf ),
    environment( env ),
    stream( buf )
{}

Parser::~Parser() {}

bool
Parser::evaluateNext() {
    if( stream.eof() )
        return false;

    char buffer[MAX_COMMAND_LEN];
    stream.getline( buffer, MAX_COMMAND_LEN );
    buffer[strlen(buffer)] ='\0';

    return evaluate( tokenize( buffer ) );
}

bool 
Parser::evaluate( stringlist_t tokens ) {
    if( tokens.size() == 0 )
        return true;

    std::string command =*tokens.begin();
    tokens.erase( tokens.begin() );

    if( command == std::string("quit") )
        return false;

    std::cout << command << std::endl;

    return true;
}

stringlist_t
Parser::tokenize( char *buffer ) {
    stringlist_t tokenlist;
    char *bufptr = buffer;
    char *token;
    while( ( token = tokennext( &bufptr ) ) ) {
        char *str;
        size_t len =tokenlen( token );
        if( !len )
            break;
        str =new char[len];
        tokencpy( str, token );
        tokenlist.push_back( std::string( str, len ) );
        delete str;
    }

    return tokenlist;
}

size_t 
Parser::tokenlen (const char *token ) {
    size_t len =0;
    while( *token != '\0' )
    {
        if( *token != 1 )
            len++;
            token++;
    }
    return len;
}

size_t 
Parser::tokencpy( char *dst, const char* src ) {
    size_t len =0;

    do {
        if( *src == 1 )
            continue;
        dst[len] = *src;
        len++;
    } while( *src++ != '\0' );
    return len;
}

char* 
Parser::tokennext( char** buffer ) {
    bool escape =false;
    bool within_quote =false;
    /* Skip leading whitespace */
    while( **buffer == ' ' ) (*buffer)++;
    char *begin =*buffer;

    while( **buffer != '\0' ) {
        if( **buffer == '\\' && !escape ) {
            escape =true;
            goto end;
        }
        if( **buffer == '"') {
            if( escape )
                *(*buffer-1) =1;
             else {
                within_quote = !within_quote;
                **buffer =1;
             }
        }

        if( **buffer == ' ' && !within_quote) {
            if( escape )
                *(*buffer-1) =1;
            else {
                **buffer ='\0';
                (*buffer)++;
                break;
            }
        } 

        escape =false;
        end:
        (*buffer)++;
    }

    return begin;
}
