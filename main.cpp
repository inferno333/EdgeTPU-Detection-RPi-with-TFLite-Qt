#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include <QtDebug>
#include <QFile>
#include <QTextStream>

#include "tensorflow.h"
#include "auxutils.h"
#include "objectsrecogfilter.h"

double AuxUtils::angleHor = 0;
double AuxUtils::angleVer = 0;
int    AuxUtils::width    = 0;
int    AuxUtils::height   = 0;

using namespace tflite;

int main(