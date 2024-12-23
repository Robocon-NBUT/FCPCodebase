#pragma once

template <class T>
class Singleton {
public:

    static T& getInstance() {
        static T instance;
        return instance;
    }

private:
    Singleton();
    ~Singleton();
    Singleton(Singleton const&);
    Singleton& operator=(Singleton const&);
};

