TEMPLATE = lib
TARGET = neural

CONFIG -= qt

CONFIG += warn_on
QMAKE_CXXFLAGS += -Werror -Wextra -pedantic-errors
QMAKE_CXXFLAGS += -std=c++14

DESTDIR = $$PWD/build/lib
OBJECTS_DIR = $$PWD/.o

HEADERS = neural.h

SOURCES = neural.cpp

target.path = $$PREFIX/lib
includes.path = $$PREFIX/include
includes.files = $$HEADERS

INSTALLS += target includes
