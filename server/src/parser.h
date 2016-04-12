#pragma once

#include <streambuf>
#include <iostream>
#include "object.h"

#define DELIM_CHARS " ,;"
#define POPEN_CHARS "(<"
#define PCLOSE_CHARS ")>"
#define MEMBER_CHARS ".:/"
#define STRESCAPE_CHARS "\"'"

class Token {
    public:
        enum TOKEN {
            TOKEN_NONE,
            TOKEN_ID,
            TOKEN_STRING,
            TOKEN_REAL,
            TOKEN_POPEN,
            TOKEN_PCLOSE
        };
        Token( TOKEN t = TOKEN_NONE ) { type = t; }
        Token( TOKEN t, std::string str ) { type = t, value_string =str; }

        TOKEN type;
        std::string value_string;
        double value_real;

        static Token fromString( const std::string& );
        typedef std::vector<Token> list;
};

class Symbol {
    public:
        typedef std::vector<Symbol> list;
        enum SYM {
            SYM_NONE,
            SYM_STRING,
            SYM_REAL,
            SYM_OBJECT, /* pointer to object */
            SYM_OBJECT_MEMBER, /* pointer to object and member variable */
            SYM_ID /* named reference */
        };
        Symbol( SYM s = SYM_NONE ) { type = s; }

        SYM type;
        std::string value_string;
        double value_real;
        Object* value_ptr;
};


class Statement {
    public:
        enum TYPE {
            TYPE_NONE,
            TYPE_ERROR,
            TYPE_STMNT, /* children depending on statement type */
            TYPE_SYMBOL, /* no children */
            TYPE_ARGLIST /* left-child: symbol, right-child: next argument */
        };
        enum STMNT {
            /* left-child: none, right-child: none */
            STMNT_NONE,
            /* left-child: none, right-child: none */
            STMNT_END,
            /* left-child: symbol (object-member), right-child: symbol */
            STMNT_SET,
            /* left-child: symbol (id), right-child: symbol (class name) */
            STMNT_NEW,
            /* left-child: symbol (object), right-child: none*/
            STMNT_DELETE,
            /* left-child: optional symbol, right-child: none */
            STMNT_LIST,
            /* left-child: symbol (object-member), right-child: none */
            STMNT_GET,
            /* left-child: none, right-child: none */
            STMNT_CLASSLIST,
            /* left-child: symbol (object-member), right-child: arglist */
            STMNT_CALL
        };
        Statement( TYPE t ) : leftChild(0), rightChild(0) { type =t; };
        Statement( STMNT s ) : leftChild(0), rightChild(0){ type =TYPE_STMNT; statement =s; }
        Statement( ): leftChild(0), rightChild(0){ type = TYPE_NONE; }
        ~Statement() { if( leftChild ) delete leftChild; if( rightChild ) delete rightChild; }

        TYPE type;
        STMNT statement;
        Symbol symbol;
        Statement *leftChild;
        Statement *rightChild;
        std::string detail;
};

class Parser {
    public:
        Parser( std::basic_streambuf<char>* buffer, Object *scope =0 );
        ~Parser();

        void setScope( Object* s ) { scope =s; }

        bool evaluateNext();
        bool evaluate( Statement* );
        Statement* parse( const Token::list& );

    private:
        std::basic_streambuf<char>* iobuf;
        Object *scope;
        std::iostream stream;

        Statement* parseArglist( const Token::list& );

        static Token::list tokenize( const std::string& );
};
