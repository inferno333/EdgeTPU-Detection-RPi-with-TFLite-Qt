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

double ObjectsRecogFilter::getCameraOrientation()
{
    return camOrientation;
}

double ObjectsRecogFilter::getVideoOrientation()
{
    return vidOrientation;
}

bool ObjectsRecogFilter::getRunning()
{
    QMutexLocker locker(&mutex);

    bool val = running;
    if (!val) setRunning(true);

    return !val;
}

void ObjectsRecogFilter::setRunning(bool val)
{
    running = val;
}

bool ObjectsRecogFilter::getInitialized() const
{
    return initialized;
}

void ObjectsRecogFilter::setInitialized(bool value)
{
    initialized = value;
}

void ObjectsRecogFilter::releaseRunning()
{
    QMutexLocker locker(&mutex);

    setRunning(false);
}

QSize ObjectsRecogFilter::getContentSize() const
{
    return videoSize;
}

void ObjectsRecogFilter::setContentSize(const QSize &value)
{
    videoSize = value;
}

bool ObjectsRecogFilter::getAcceleration() const
{
    return acc;
}

void ObjectsRecogFilter::setAcceleration(bool value)
{
    acc = value;
}

int ObjectsRecogFilter::getNThreads() const
{
    return nThr;
}

void ObjectsRecogFilter::setNThreads(int value)
{
    nThr = value;
}

bool ObjectsRecogFilter::getShowInfTime() const
{
    return infTime;
}

void ObjectsRecogFilter::setShowInfTime(bool value)
{
    infTime = value;
}

double ObjectsRecogFilter::getMinConfidence() const
{
    return minConf;
}

void ObjectsRecogFilter::setMinConfidence(double value)
{
    minConf = value;
    tf.setThreshold(minConf);
}

ObjectsRecogFilterRunable::ObjectsRecogFilterRunable(ObjectsRecogFilter *filter, QStringList res)
{
    m_filter   = filter;
    results    = res;
}

void ObjectsRecogFilterRunable::setResults(int net, QStringList res, QList<double> conf, QList<QRectF> box, QList<QImage> mask, int inftime)
{
    network       = net;
    results       = res;
    confidence    = conf;
    boxes         = box;
    masks         = mask;
    inferenceTime = inftime;
}

void ObjectsRecogFilter::setActiveLabel(QString key, bool value)
{
    activeLabels[key] = value;
}

QMap<QString,bool> ObjectsRecogFilter::getActiveLabels()
{
    return activeLabels;
}

bool ObjectsRecogFilter::getActiveLabel(QString key)
{
    return activeLabels.value(key,false);
}

double ObjectsRecogFilter::getAngle() const
{
    return ang;
}

void ObjectsRecogFilter::setAngle(const double value)
{
    ang = value;    
    emit angleChanged();
}

double ObjectsRecogFilter::getImgHeight()
{
    return tf.getHeight();
}

double ObjectsRecogFilter::getImgWidth()
{
    return tf.getWidth();
}

QImage rotateImage(QImage img, double rotation)
{
    QPoint center = img.rect().center();
    QMatrix matrix;
    matrix.translate(center.x(), center.y());
    matrix.rotate(rotation);

    return img.transformed(matrix);
}

QVideoFrame ObjectsRecogFilterRunable::run(QVideoFrame *input, const QVideoSurfaceFormat &surfaceFormat, RunFlags flags)
{
    Q_UNUSED(surfaceFormat);
    Q_UNUSED(flags);

    QImage img;
    bool mirrorHorizontal;
    bool mirrorVertical = false;

    if(input->isValid())
    {       
        // Get image from video frame, we need to convert it
        // for unsupported QImage formats, i.e Format_YUV420P
        //
        // When input has an unsupported format the QImage
        // default format is ARGB32
        //
        // NOTE: BGR im