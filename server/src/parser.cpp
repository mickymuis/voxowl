#include "parser.h"
#include <string.h>

Token
Token::fromString( const std::string& str ) {
    Token t;
    char *p;
    double d =strtod( str.c_str(), &p );
    if( *p == 0 ) {
        t.type =TOKEN_REAL;
        t.value_real =d;
    }
    else if( str.size() == 1 && std::string( POPEN_CHARS ).find( str[0] ) != std::string::npos )
        t.type =TOKEN_POPEN;
    else if( str.size() == 1 && std::string( PCLOSE_CHARS ).find( str[0] ) != std::string::npos )
        t.type =TOKEN_PCLOSE;
    else if( str.size() > 1 && std::string( STRESCAPE_CHARS ).find( str[0] ) != std::string::npos
            && std::string( STRESCAPE_CHARS ).find( str[str.size()-1] ) != std::string::npos ) {
        t.type =TOKEN_STRING;
        t.value_string =str.substr( 1, str.size()-2 );
    }
    else {
        t.type =TOKEN_ID;
        t.value_string =str;
    }
    return t;
}

Parser::Parser( std::basic_streambuf<char> *buf, Object* s )
    : iobuf( buf ),
    scope( s ),
    stream( buf )
{}

Parser::~Parser() {}

bool
Parser::evaluateNext() {
    if( stream.eof() )
        return false;

    std::string buffer;
    std::getline( stream, buffer );
    buffer.resize( buffer.size() -1);
    std::cout << buffer << std::endl;

    bool b;
    Statement* s;

    s =parse( tokenize( buffer ) );
    b =evaluate( s );

    delete s;
    return b;
}

bool 
Parser::evaluate( Statement *s ) {

    switch( s->type ) {
        case Statement::TYPE_NONE:
            break;
        case Statement::TYPE_ERROR:
            stream << "Error: " << s->detail << std::endl;
            break;
        case Statement::TYPE_STMNT:
            switch( s->statement ) {

                /* left-child: none, right-child: none */
                case Statement::STMNT_NONE:
                    break;

                /* left-child: none, right-child: none */
                case Statement::STMNT_END:
                    return false;

                /* left-child: symbol (object-member), right-child: symbol */
                case Statement::STMNT_SET:
                    break;
                /* left-child: symbol (id), right-child: symbol (class name) */
                case Statement::STMNT_NEW:
                    break;
                /* left-child: symbol (object), right-child: none*/
                case Statement::STMNT_DELETE:
                    break;
                /* left-child: optional symbol, right-child: none */
                case Statement::STMNT_LIST:
                    break;
                /* left-child: symbol (object-member), right-child: none */
                case Statement::STMNT_GET:
                    break;
                /* left-child: symbol (object-member), right-child: arglist */
                case Statement::STMNT_CALL:
                    break;
                /* left-child: none, right-child: none */
                case Statement::STMNT_CLASSLIST: {
                        int count =ObjectFactory::getFactoryCount();
                        for( int i=0; i < count; i++ )
                            stream << "class " << ObjectFactory::getFactory(i)->getName() << ";" << std::endl;
                    break;
                }
            }
            break;
        case Statement::TYPE_SYMBOL:
            break;
        case Statement::TYPE_ARGLIST:
            break;
    }

    return true;
}

Statement*
Parser::parse( const Token::list& tokenlist ) {
    Statement *s;

    if( tokenlist[0].type != Token::TOKEN_ID ) {
        s = new Statement( Statement::TYPE_ERROR );
        s->detail ="Expected statement or ID";
    }
    else {
        if( tokenlist[0].value_string == "quit" )
            s = new Statement( Statement::STMNT_END );
        else if( tokenlist[0].value_string == "set" ) {
            s = new Statement( Statement::STMNT_SET );
        }
        else if( tokenlist[0].value_string == "get" ) {
            s = new Statement( Statement::STMNT_GET );
        }
        else if( tokenlist[0].value_string == "new" ) {
            s = new Statement( Statement::STMNT_NEW );
        }
        else if( tokenlist[0].value_string == "delete" ) {
            s = new Statement( Statement::STMNT_DELETE );
        }
        else if( tokenlist[0].value_string == "list" ) {
            if( tokenlist.size() != 1 || ( tokenlist.size() > 1 && tokenlist[1].type != Token::TOKEN_ID ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too many arguments to `list'";
            } else {
                s = new Statement( Statement::STMNT_LIST );
                s->leftChild = new Statement( Statement::TYPE_SYMBOL );
                s->leftChild->symbol = Symbol( Symbol::SYM_ID, tokenlist[1].value_string );
            }
        }
        else if( tokenlist[0].value_string == "classlist" ) {
            s = new Statement( Statement::STMNT_CLASSLIST );
        }
        else {
            if( tokenlist.size() > 2 && tokenlist[1].type == Token::TOKEN_POPEN
                    && tokenlist[tokenlist.size()-1].type == Token::TOKEN_PCLOSE ) {
                s = new Statement( Statement::STMNT_CALL );
            }
            else {
                s = new Statement( Statement::STMNT_GET );
            }
        }
    }

    /*for( unsigned int i=0; i < tokenlist.size(); i++ ) {
        switch( tokenlist[i].type ) {
            case Token::TOKEN_ID:
                stream << "ID: " << tokenlist[i].value_string << std::endl;
                break;
            case Token::TOKEN_STRING:
                stream << "STRING: " << tokenlist[i].value_string << std::endl;
                break;
            case Token::TOKEN_REAL:
                stream << "REAL: " << tokenlist[i].value_real << std::endl;
                break;
            case Token::TOKEN_POPEN:
                stream << "POPEN: " << std::endl;
                break;
            case Token::TOKEN_PCLOSE:
                stream << "PCLOSE: " << std::endl;
                break;
            default:
                stream << "?" << std::endl;
        }
    }*/

    return s;
}

Statement*
Parser::parseReference( const std::string& str ) {

}

Statement*
Parser::parseArglist( const Token::list& ) {

}

Token::list
Parser::tokenize( const std::string& buffer ) {
    static const std::string paren( std::string( POPEN_CHARS ) + std::string( PCLOSE_CHARS ) );
    Token::list tokenlist;

    bool escape =false;
    bool within_quote =false;
    unsigned int j;
    unsigned int i = j = buffer.find_first_not_of( ' ' );
    std::string str;

    while( j < buffer.size() ) {

        if( !within_quote && !escape && 
                ( std::string( DELIM_CHARS ).find( buffer[j] ) != std::string::npos 
                || paren.find( buffer[j] ) != std::string::npos ) ) {
            if( j-i ) {
                str += buffer.substr( i, j-i );
                tokenlist.push_back( Token::fromString(str) );
                str = std::string();
            }
            if( paren.find( buffer[j] ) != std::string::npos )
                tokenlist.push_back( Token::fromString( std::string( &buffer[j], 1 ) ) );
            i = j = buffer.find_first_not_of( ' ', j );
        }
        else if( !escape && std::string( STRESCAPE_CHARS).find( buffer[j] ) != std::string::npos ) {
            str += buffer.substr( i, j-i );
            if( !str.empty() ) {
                if( within_quote )
                    tokenlist.push_back( Token( Token::TOKEN_STRING, str ) );
                else
                    tokenlist.push_back( Token::fromString(str) );

                str = std::string();
            }
            within_quote = !within_quote;
            i = ++j;
        }
        else if( !escape && buffer[j] == '\\' ) {
            escape =true;
            str += buffer.substr( i, j-i );
            i = ++j;
        }
        else {
            j++;
            escape =false;
        }
    }
    if( j-i )
        str += buffer.substr( i, j-i );
    if( !str.empty() ){
        tokenlist.push_back( Token::fromString(str) );
    }
    return tokenlist;
}

