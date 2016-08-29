#include "performance_counter.h"

void PerformanceCounter::update( float value ) {
    if( value < m_min || !m_count )
        m_min =value;
    if( value > m_max )
        m_max =value;
    m_mean =m_mean * ((float)m_count / (float)++m_count) + value / (float)m_count;
}

void PerformanceCounter::reset() {
    m_count =0;
    m_mean =m_min =m_max =0.f;
}

PerformanceCounter *PerformanceCounter::create( const std::string& key, const std::string& description, const std::string& unit_name ) {
    PerformanceCounter *p =find( key );
    if( p ) return p;
    p = new PerformanceCounter( key, description, unit_name );
    counters[key] =p;
    return p;
}

void PerformanceCounter::prettyPrint( std::ostream& str ) const {
    str << m_description << " (N=" << m_count << ")" << std::endl
        << "\tmean: " << m_mean << " " << m_unit_name << std::endl
        << "\t min: "  << m_min  << " " << m_unit_name << std::endl
        << "\t max: "  << m_max  << " " << m_unit_name << std::endl;
}

PerformanceCounter *PerformanceCounter::find( const std::string& key ) {
    map_t::iterator it =counters.find( key );
    if( it == counters.end() )
        return 0;
    return it->second;
}

void PerformanceCounter::update( const std::string& key, float value ) {
    map_t::iterator it =counters.find( key );
    if( it == counters.end() )
        return;
    it->second->update( value );
}

void PerformanceCounter::printAll( std::ostream& str ) {
    map_t::iterator it;
    for( it = counters.begin(); it != counters.end(); it++ )
        it->second->prettyPrint( str );

}

void PerformanceCounter::resetAll( ) {
    map_t::iterator it;
    for( it = counters.begin(); it != counters.end(); it++ )
        it->second->reset();

}

void PerformanceCounter::cleanup() {
    map_t::iterator it;
    for( it = counters.begin(); it != counters.end(); it++ ) {
        delete it->second;
    }
}

PerformanceCounter::~PerformanceCounter() {
    counters.erase( m_key );
}

PerformanceCounter::map_t PerformanceCounter::counters;
