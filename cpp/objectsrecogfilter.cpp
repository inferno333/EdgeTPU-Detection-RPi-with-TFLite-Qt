#include <objectsrecogfilter.h>
#include <QStandardPaths>
#include <QPainter>
#include <QDebug>
#include <QThread>
#include <QMutexLocker>

#include "auxutils.h"
#include "private/qvideoframe_p.h"

// WARNING: same TensorFlow initialization repeated in ObjectRecogFilter and TensorFlowQML constructors
ObjectsRecogFilter::ObjectsRecogFilter()
{
    connect(this, SIGNAL(runTensorFlow(QImage)), this, SLOT(TensorFlowExecution(QImage)));
    connect(&tft,SIGNAL(results(int, QStringList, QList<double>, QList<QRectF>, QList<QImage>, int)),this,SLOT(processResults(int, QStringList, QList<double>, QList<QRectF>, QList<QImage>, int)));

    tf.setFilename(AuxUtils::getDefaultModelFilename());
    tf.setLabelsFilename(AuxUtils::getDefaultLabelsFilename());
    tf.setAccelaration(true);
    tf.setNumThreads(QThread::idealThreadCount());

    releaseRunning();
    initialized = false;
}

void ObjectsRecogFilter::init(int imgHeight, int imgWidth)
{
    initialized = tf.init(imgHeight,imgWidth);
    tft.setTf(&tf);
}

void ObjectsRecogFilter::initInput(int imgHeight, int imgWidth)
{
    tf.initInput(imgHeight,imgWidth);
}

void ObjectsRecogFilter::TensorFlowExecution(QImage imgTF)
{
    tf.setAccelaration(getAcceleration());
    tf.setNumThreads(getNThreads());
    tft.run(imgTF);
}

void ObjectsRecogFilter::processResults(int network, QStringList res, QList<double> conf, QList<QRectF> boxes, QList<QImage> masks, int inftime)
{
    rfr->setResults(network,res,conf,boxes,masks,inftime);
    releaseRunning();
}

void ObjectsRecogFilter::setCameraOrientation(double o)
{
    camOrientation = o;
}

void ObjectsRecogFilter::setVideoOrientation(double o)
{
    vidOrientation = o;
}

double Objec