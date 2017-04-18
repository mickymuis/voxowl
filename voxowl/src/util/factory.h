
#include <map>

template<class BaseT>
class Creator {
    public:
        virtual BaseT* create() =0;
};

template<class BaseT, class DerivedT>
class DerivedCreator : public Creator<BaseT> {
    public:
        virtual BaseT* create() { return new DerivedT(); }
};

template<class BaseT, class KeyT>
class Factory {
    public:
        Factory() {}
        ~Factory();
        typedef std::map<KeyT, Creator<BaseT>*> map_t;

        void registerCreator( const KeyT& key, Creator<BaseT>* c ) {
            map[key] =c;
        }

        BaseT* create( const KeyT& key );
        const map_t& getMap() const { return map; }

    protected:
        map_t map;

};

template<class BaseT, class KeyT>
Factory<BaseT,KeyT>::~Factory() {
    for( const auto& pair : map )
        delete pair.second;
}

template<class BaseT, class KeyT>
BaseT*
Factory<BaseT,KeyT>::create( const KeyT& key ) {
    if( map.find( key ) == map.end() )
        return nullptr;
    return map[key]->create();
}

