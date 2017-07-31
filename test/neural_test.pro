TEMPLATE = app
TARGET = test

CONFIG -= qt

CONFIG += warn_on
QMAKE_CXXFLAGS += -Werror -Wextra -pedantic-errors
QMAKE_CXXFLAGS += -std=c++14

DESTDIR = $$PWD/build/bin
OBJECTS_DIR = $$PWD/.o

SOURCES = main.cpp

target.path = $$PREFIX/bin
INSTALLS += target

INCLUDEPATH += $$PREFIX/include

LIBS += -L$$PREFIX/lib -lneural
