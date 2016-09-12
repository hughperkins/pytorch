#include <iostream>
#include <stdexcept>
using namespace std;

#ifndef _WIN32
    #include <dlfcn.h>
#endif

extern "C" {
  void loadLuajit();
  void initLuajitLoader();
}

void loadLuajit() {
    void *hdl = dlopen("libluajit.so", RTLD_NOW | RTLD_GLOBAL);
    if(hdl == 0) {
        hdl = dlopen("libluajit.dylib", RTLD_NOW | RTLD_GLOBAL);
    }
    if(hdl == 0) {
        cout << "Failed to load both libluajit.so and libluajit.dylib, fatal" << endl;
        cout << dlerror() << endl;
        throw runtime_error(string("Couldnt load libluajit.so or libluajit.dylib") + dlerror());
    } else {
    }
}

void initLuajitLoader() {
  loadLuajit();
}

