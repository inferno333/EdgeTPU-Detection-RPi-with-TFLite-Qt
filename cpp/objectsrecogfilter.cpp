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
    initialized = f