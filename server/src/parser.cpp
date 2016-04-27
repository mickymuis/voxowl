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
                case Statement::STMNT_LIST: {
                    if( s->leftChild && s->leftChild->type == Statement::TYPE_ERROR ) {
                        stream << "Error: " << s->leftChild->detail << std::endl;
                        break;
                    }
                    Object *obj =scope;
                    if( s->leftChild && s->leftChild->symbol.type == Symbol::SYM_OBJECT )
                        obj =s->leftChild->symbol.value_ptr;

                    generateManifest( stream, obj );
                    break;
                }
                /* left-child: symbol (object-member), right-child: none */
                case Statement::STMNT_GET: {
                    if( s->leftChild && s->leftChild->type == Statement::TYPE_ERROR ) {
                        stream << "Error: " << s->leftChild->detail << std::endl;
                        break;
                    }
                    Object *obj;
                    obj =s->leftChild->symbol.value_ptr;
                    std::string prop = s->leftChild->symbol.value_string;
                    if( !obj->hasMeta( Object::META_PROPERTY, prop ) ) {
                        stream << "Error: no property `" << prop << "' in `" << obj->getName() << "'" << std::endl;
                        break;
                    }
                    stream << obj->getMeta( prop ).toString() << std::endl;
                    break;
                }
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
            if( tokenlist.size() > 2 ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too many arguments to `get'";
            } else if( tokenlist.size() < 2 ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too few arguments to `get'";
            } else if( tokenlist[1].type != Token::TOKEN_ID ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Expected reference in `" + tokenlist[1].value_string + "'";
            } else  {
                s = new Statement( Statement::STMNT_GET );
                /* Only the global scope is currently supported */
                s->leftChild =parseMemberReference( scope, tokenlist[1].value_string );
            }
        }
        else if( tokenlist[0].value_string == "new" ) {
            s = new Statement( Statement::STMNT_NEW );
        }
        else if( tokenlist[0].value_string == "delete" ) {
            s = new Statement( Statement::STMNT_DELETE );
        }
        else if( tokenlist[0].value_string == "list" ) {
            if( tokenlist.size() > 2 || ( tokenlist.size() == 2 && tokenlist[1].type != Token::TOKEN_ID ) ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too many arguments to `list'";
            } else {
                s = new Statement( Statement::STMNT_LIST );
                /* Only the global scope is currently supported */
                if( tokenlist.size() > 1 )
                    s->leftChild =parseReference( scope, tokenlist[1].value_string );
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
Parser::parseReference( Object* local_scope, const std::string& str ) {
    if( str.empty() )
        return 0;
    Statement *s;
    Object *obj = local_scope->getChildByName( str );
    if( obj ) {
        s = new Statement( Statement::TYPE_SYMBOL );
        s->symbol = Symbol( obj );
    }
    else {
        s = new Statement( Statement::TYPE_ERROR );
        s->detail = "Undefined reference to `" + str + "'";
    }
    return s;
}

Statement*
Parser::parseMemberReference( Object* local_scope, const std::string& str ) {
    if( str.empty() )
        return 0;
    Statement *s;
    size_t pos = str.find_last_of( MEMBER_CHARS );
    std::string member;
    Object *obj;
    if( pos == std::string::npos ) {
        member =str;
        obj =local_scope;
    }
    else {
        obj = local_scope->getChildByName( str.substr( 0, pos ) );
        member =str.substr( pos+1 );
    }

    if( obj ) {
        s = new Statement( Statement::TYPE_SYMBOL );
        s->symbol = Symbol( obj, member );
    }
    else {
        s = new Statement( Statement::TYPE_ERROR );
        s->detail = "Undefined reference to `" + str + "'";
    }
    return s;
}

Statement*
Parser::parseArglist( const Token::list& ) {

}

void
Parser::generateManifest( std::iostream &out, Object* obj ) {

    for( int k =1; k <= 3; k++ ) {
        stringlist_t list =obj->listMeta( (Object::META_TYPE)k );
        if( list.empty() )
            continue;
        std::string suffix;
        switch( k ) {
            case Object::META_CHILD:
                out << "children:" << std::endl;
                break;
            case Object::META_METHOD:
                out << "methods:" << std::endl;
                suffix ="()";
                break;
            case Object::META_PROPERTY:
                out << "properties:" << std::endl;
                break;
        }
        stringlist_t::iterator it;
        for( it = list.begin(); it != list.end(); it++ )
            out << "\t" << obj->getName() << "." << *it << suffix << std::endl;
    }
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

