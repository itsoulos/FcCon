TEMPLATE = app
QT -= gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += c++17 console
CONFIG -= app_bundle
QMAKE_CXXFLAGS_RELEASE += -O4 -fopenmp -march=native -unroll-loops -omit-frame-pointer -Winline -unsafe-math-optimizations -mtune=native  -ffast-math -fopt-info -Ofast
QMAKE_CFLAGS_RELEASE += -O4 -fopenmp -march=native -unroll-loops -omit-frame-pointer -Winline -unsafe-math-optimizations -mtune=native -ffast-math -Ofast

QMAKE_CXXFLAGS_RELEASE += -std=c++11
QMAKE_CFLAGS_RELEASE += -std=c++11

QMAKE_CXXFLAGS += -O4 -fopenmp -march=native -unroll-loops -omit-frame-pointer -Winline -unsafe-math-optimizations -mtune=native -ffast-math -Ofast
QMAKE_CFLAGS += -O4 -fopenmp -march=native -unroll-loops -omit-frame-pointer -Winline -unsafe-math-optimizations -mtune=native  -ffast-math -Ofast
SOURCES += \
        CORE/collection.cc \
        CORE/parameter.cpp \
        CORE/parameterlist.cpp \
        CORE/problem.cc \
        GE/cprogram.cc \
        GE/doublestack.cc \
        GE/fparser.cc \
        GE/fpoptimizer.cc \
        GE/nnprogram.cc \
        GE/population.cc \
        GE/program.cc \
        GE/rule.cc \
        GE/symbol.cc \
        GRS/grs.cc \
        GRS/grspopulation.cc \
        GRS/rlsprogram.cc \
        MLMODELS/Rbf.cc \
        MLMODELS/gensolver.cc \
        MLMODELS/kmeans.cc \
        MLMODELS/knn.cc \
        MLMODELS/mapper.cc \
        MLMODELS/matrix_functions.cc \
        MLMODELS/model.cc \
        MLMODELS/neural.cc \
        MLMODELS/rbf_model.cc \
        MLMODELS/tolmin.cc \
        main.cpp

HEADERS += \
    CORE/collection.h \
    CORE/parameter.h \
    CORE/parameterlist.h \
    CORE/problem.h \
    GE/cprogram.h \
    GE/doublestack.h \
    GE/f2c.h \
    GE/fparser.hh \
    GE/fpconfig.hh \
    GE/fptypes.hh \
    GE/nnprogram.h \
    GE/population.h \
    GE/program.h \
    GE/rule.h \
    GE/symbol.h \
    GRS/grs.h \
    GRS/grspopulation.h \
    GRS/rlsprogram.h \
    MLMODELS/Rbf.h \
    MLMODELS/gensolver.h \
    MLMODELS/kmeans.h \
    MLMODELS/knn.h \
    MLMODELS/mapper.h \
    MLMODELS/matrix_functions.h \
    MLMODELS/model.h \
    MLMODELS/neural.h \
    MLMODELS/rbf_model.h
