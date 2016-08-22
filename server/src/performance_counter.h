#pragma once

#include <iostream>
#include <map>
#include <string>

/*! Performance counter that performs some simple statistics */

class PerformanceCounter {
    public:
        typedef std::map<std::string, PerformanceCounter*> map_t;

        void update( float value );
        void reset();

        inline int count() const { return m_count; }
        inline float mean() const { return m_mean; }
        inline float min() const { return m_min; }
        inline float man() const { return m_max; }
        inline std::string unitName() const { return m_unit_name; }
        inline std::string description() const { return m_description; }

        void prettyPrint( std::ostream& ) const;

    public: // static
        static PerformanceCounter *create( const std::string& key, const std::string& description, const std::string& unit_name );

        static PerformanceCounter *find( const std::string& key );
        static void update( const std::string& key, float value );

        static void printAll( std::ostream& );

        static void cleanup();

    protected:
        inline PerformanceCounter( const std::string& key, const std::string& description, const std::string& unit_name );
        ~PerformanceCounter();

        static map_t counters;

    private:
        int m_count;
        float m_mean;
        float m_min, m_max;
        std::string m_key;
        std::string m_description;
        std::string m_unit_name;
};

inline PerformanceCounter::PerformanceCounter( const std::string& key, const std::string& description, const std::string& unit_name ) 
    : m_key( key ), m_description( description ), m_unit_name( unit_name ) { reset(); }

