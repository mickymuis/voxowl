#include "parser.h"
#include <string.h>
#include <sstream>

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

    bool last =false, error;
    Variant v;
    Statement* s;

    s =parse( tokenize( buffer ) );
    if( s ) {
        error =evaluate( v, last, s );
        stream << v << std::endl;
        delete s;
    }
    return !last;
}

bool /*error*/ 
Parser::evaluate( Variant& result, bool &last, Statement *s ) {
    bool error = false;
    last = false;
    std::stringstream out_string;

    stringlist_t errs;
    if( listErrors( errs, s ) ) {
        error =true;
        std::string err_str;
        for( unsigned int i=0; i < errs.size(); i++ )
            out_string << errs[i] << std::endl;
    }
    else {
        switch( s->type ) {
            case Statement::TYPE_STMNT:
                switch( s->statement ) {

                    /* left-child: none, right-child: none */
                    case Statement::STMNT_NONE:
                        break;
                    /* left-child: none, right-child: none */
                    case Statement::STMNT_END:
                        last =true;
                        break;
                    /* left-child: symbol (object-member), right-child: symbol */
                    case Statement::STMNT_SET: {
                        Object *obj =s->leftChild->symbol.value_ptr;
                        std::string prop =s->leftChild->symbol.value_string;
                        Variant::list args;
                        dereferenceArglist( args, s->rightChild );
                        if( args.size() )
                            obj->setMeta( prop, args[0] );

                        break;
                    }
                    /* left-child: symbol (id/classname), right-child: symbol (object-member) */
                    case Statement::STMNT_NEW: {
                        std::string class_name =s->leftChild->symbol.value_string;
                        ObjectFactory *fac =ObjectFactory::getFactoryByName( class_name );
                        if( !fac ) {
                            error =true;
                            out_string << "Error: `" << class_name << "' is not a class or type" << std::endl;
                            break;
                        }
                        std::string instance_name =s->rightChild->symbol.value_string;
                        Object *parent =s->rightChild->symbol.value_ptr;
                        fac->create( instance_name, parent );
                        break;
                    }
                    /* left-child: symbol (object), right-child: none*/
                    case Statement::STMNT_DELETE: {
                        Object *obj =s->leftChild->symbol.value_ptr;
                        if( obj != scope )
                            delete obj;
                        break;
                    }
                    /* left-child: optional symbol, right-child: none */
                    case Statement::STMNT_LIST: {
                        Object *obj =scope;
                        if( s->leftChild && s->leftChild->symbol.type == Symbol::SYM_OBJECT )
                            obj =s->leftChild->symbol.value_ptr;

                        generateManifest( out_string, obj );
                        break;
                    }
                    /* left-child: symbol (object-member), right-child: none */
                    case Statement::STMNT_GET: {
                        Object *obj;
                        obj =s->leftChild->symbol.value_ptr;
                        std::string prop = s->leftChild->symbol.value_string;
                        if( !obj->hasMeta( Object::META_PROPERTY, prop ) ) {
                            error =true;
                            out_string << "Error: no property `" << prop << "' in `" << obj->getName() << "'" << std::endl;
                            break;
                        }
                        result = obj->getMeta( prop ).toString();
                        break;
                    }
                    /* left-child: symbol (object-member), right-child: arglist */
                    case Statement::STMNT_CALL: {
                        Variant::list args;
                        dereferenceArglist( args, s->rightChild );
                        result =functioncall( s->leftChild->symbol, args );
                        break;
                    }
                    /* left-child: none, right-child: none */
                    case Statement::STMNT_CLASSLIST: {
                            int count =ObjectFactory::getFactoryCount();
                            for( int i=0; i < count; i++ )
                                out_string << "class " << ObjectFactory::getFactory(i)->getName() << ";" << std::endl;
                        break;
                    }
                }
                break;
            default:
                break;
        }
    }

    if( !out_string.str().empty() )
        result.set( out_string.str() );

    return error;
}

Statement*
Parser::parse( const Token::list& tokenlist ) 
{
/*    for( unsigned int i=0; i < tokenlist.size(); i++ ) {
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
    if( tokenlist.size() == 0 )
        return 0;

    Statement *s;

    if( tokenlist[0].type != Token::TOKEN_ID ) {
        s = new Statement( Statement::TYPE_ERROR );
        s->detail ="Expected statement or ID";
    }
    else {
        if( tokenlist[0].value_string == "quit" )
            s = new Statement( Statement::STMNT_END );
        else if( tokenlist[0].value_string == "set" ) {
            if( tokenlist.size() < 3 ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too few arguments to `set'";
            } else if( tokenlist[1].type != Token::TOKEN_ID ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Expected reference in `" + tokenlist[1].value_string + "'";
            } else  {
                s = new Statement( Statement::STMNT_SET );
                s->leftChild =parseMemberReference( scope, tokenlist[1].value_string, Object::META_PROPERTY );
                s->rightChild =parseArglist( scope, tokenlist.begin() + 2, tokenlist.end() );
            }
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
                s->leftChild =parseMemberReference( scope, tokenlist[1].value_string, Object::META_PROPERTY );
            }
        }
        else if( tokenlist[0].value_string == "new" ) {
            if( tokenlist.size() > 3 ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too many arguments to `new'";
            } else if( tokenlist.size() < 3 ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too few arguments to `new'";
            } else if( tokenlist[1].type != Token::TOKEN_ID ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Expected identifier in `" + tokenlist[1].value_string + "'";
            } else if( tokenlist[2].type != Token::TOKEN_ID ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Expected reference in `" + tokenlist[2].value_string + "'";
            } else  {
                s = new Statement( Statement::STMNT_NEW );
                s->leftChild = new Statement( Statement::TYPE_SYMBOL );
                s->leftChild->symbol =Symbol( Symbol::SYM_ID );
                s->leftChild->symbol.value_string =tokenlist[1].value_string;
                s->rightChild =parseMemberReference( scope, tokenlist[2].value_string, Object::META_ALL, false );
            }
        }
        else if( tokenlist[0].value_string == "delete" ) {
            if( tokenlist.size() > 2 ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too many arguments to `delete'";
            } else if( tokenlist.size() < 2 ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Too few arguments to `delete'";
            } else if( tokenlist[1].type != Token::TOKEN_ID ) {
                s = new Statement( Statement::TYPE_ERROR );
                s->detail = "Expected reference in `" + tokenlist[1].value_string + "'";
            } else  {
                s = new Statement( Statement::STMNT_DELETE );
                s->leftChild =parseReference( scope, tokenlist[1].value_string );
            }
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
        else if( tokenlist.size() > 2 && tokenlist[1].type == Token::TOKEN_POPEN
                    && tokenlist[tokenlist.size()-1].type == Token::TOKEN_PCLOSE ) {
            s = new Statement( Statement::STMNT_CALL );
            s->leftChild =parseMemberReference( scope, tokenlist[0].value_string );
            s->rightChild =parseArglist( scope, tokenlist.begin() + 2, tokenlist.end() - 1 );
        }
        else if( tokenlist.size() == 1 ){
            s = new Statement( Statement::STMNT_GET );
            s->leftChild =parseMemberReference( scope, tokenlist[0].value_string );
        }
        else {
            s = new Statement( Statement::TYPE_ERROR );
            s->detail = "Unrecognized statement or malformed expression";
        }
    }


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
Parser::parseMemberReference( Object* local_scope, const std::string& str, int mask, bool existence ) {
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
        bool exists = false;
        if( mask & Object::META_CHILD )
            exists |= obj->hasMeta( Object::META_CHILD, member );
        if( mask & Object::META_METHOD )
            exists |= obj->hasMeta( Object::META_METHOD, member );
        if( mask & Object::META_PROPERTY )
            exists |= obj->hasMeta( Object::META_PROPERTY, member );

        if( existence == exists ) {
            s = new Statement( Statement::TYPE_SYMBOL );
            s->symbol = Symbol( obj, member );
        } else {
            s = new Statement( Statement::TYPE_ERROR );
            if( exists )
                s->detail = "Method or property `" + str + "' exists";
            else
                s->detail = "No method or property `" + str + "'";
        }
    }
    else {
        s = new Statement( Statement::TYPE_ERROR );
        s->detail = "Undefined reference to `" + str + "'";
    }
    return s;
}

Statement*
Parser::parseArglist( Object* local_scope, const Token::list::const_iterator& begin, const Token::list::const_iterator& end, Token::list::const_iterator* last ) {

    Statement *root =0;
    Statement *s =0;
    Token::list::const_iterator it;
    for( it = begin; it != end; it++ ) {
        if( !root )
            s = root = new Statement( Statement::TYPE_ARGLIST );
        else {
            s->rightChild = new Statement( Statement::TYPE_ARGLIST );
            s = s->rightChild;
        }

        Token t =*it;
        switch( t.type ) {
            case Token::TOKEN_PCLOSE:
                // End of the argument list
                if( last )
                    *last = it;
                return root; 
            case Token::TOKEN_STRING:
                s->leftChild = new Statement( Statement::TYPE_SYMBOL );
                s->leftChild->symbol = Symbol( Symbol::SYM_STRING, t.value_string );
                break;
            case Token::TOKEN_REAL:
                s->leftChild = new Statement( Statement::TYPE_SYMBOL );
                s->leftChild->symbol = Symbol( t.value_real );
                break;
            case Token::TOKEN_ID:
                // Function call
                if( (it+1) != end && (*(it+1)).type == Token::TOKEN_POPEN ) {
                    s->leftChild = new Statement( Statement::STMNT_CALL );
                    s->leftChild->leftChild = parseMemberReference( local_scope,  t.value_string );
                    s->leftChild->rightChild = parseArglist( local_scope, it + 2, end, &it );
                }
                // Object reference
                else if( local_scope->getChildByName( t.value_string ) != 0 ) {
                    s->leftChild = parseReference( local_scope, t.value_string );
                // Property reference
                } else {
                    std::cerr << "Property reference in arglist: " << t.value_string << std::endl;
                    s->leftChild = parseMemberReference( local_scope, t.value_string, Object::META_PROPERTY );
                }
                break;
            default:
                s->type = Statement::TYPE_ERROR;
                s->detail = "Expected argument";
                break;
        }
    }
    return root;
}

void
Parser::generateManifest( std::iostream &out, Object* obj ) {

    for( int i =1; i <= 3; i++ ) {
        Object::META_TYPE k = (i>1) ? ((i==2) ? Object::META_CHILD : Object::META_METHOD) : Object::META_PROPERTY;
        stringlist_t list =obj->listMeta( k );
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
            default: break;
        }
        stringlist_t::iterator it;
        for( it = list.begin(); it != list.end(); it++ )
            out << "\t" << *it << suffix << std::endl;
    }
}

Variant
Parser::dereference( const Symbol& sym ) { 
    Variant v;
    switch( sym.type ) {
        case Symbol::SYM_REAL:
            v.set( sym.value_real );
            break;
        case Symbol::SYM_STRING:
            v.set( sym.value_string );
            break;
        case Symbol::SYM_OBJECT:
            v.set( sym.value_ptr );
            break;
        case Symbol::SYM_OBJECT_MEMBER:
            if( sym.value_ptr )
                v = sym.value_ptr->getMeta( sym.value_string );
            break;
        default: break;
    }
    return v;
}

Variant 
Parser::functioncall( const Symbol& func, const Variant::list &args ) {
    Variant v("<undefined>");
    if( func.type == Symbol::SYM_OBJECT_MEMBER && func.value_ptr ) {
        v =func.value_ptr->callMeta( func.value_string, args );
    }
    return v;
}

bool
Parser::dereferenceArglist( Variant::list& dest, Statement* s ) { 

    // Argument list-type Statement object:
    // left-child: symbol or call Statement
    // right-child: arglist-type Statement containing the next argument

    if( !s )
        return true;

    bool success =true;
    Variant v;

    if( s->type != Statement::TYPE_ARGLIST || !s->leftChild ) {
        success =false;
    } else if( s->leftChild->type == Statement::TYPE_SYMBOL ) {
        // Symbol type argument
        v = dereference( s->leftChild->symbol );
    } else if( s->leftChild->type == Statement::TYPE_STMNT && s->leftChild->statement == Statement::STMNT_CALL ) {
        // Argument is a function call
        Variant::list args; 
        dereferenceArglist( args, s->leftChild->rightChild );
        v = functioncall( s->leftChild->leftChild->symbol, args );

    } else {
        success =false;
    }

    dest.push_back( v );
    
    // Traverse the right side of the parse tree recursively to accumulate all arguments

    success &= dereferenceArglist( dest, s->rightChild );
    return success;
}

bool
Parser::listErrors( stringlist_t& dest, Statement* s ) { 

    if( !s )
        return false;

    bool error =false;

    if( s->type == Statement::TYPE_ERROR ) {
        error =true;
        dest.push_back( s->detail );
    }
    
    // Traverse the parse tree recursively to accumulate all errors

    error |= listErrors( dest, s->leftChild );
    error |= listErrors( dest, s->rightChild );
    return error;
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
            i = j = j+1;
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

