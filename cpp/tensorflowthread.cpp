#include "tensorflowthread.h"
#include "auxutils.h"

void WorkerTF::setTf(TensorFlow *value)
{
    tf = value;
}

void WorkerTF::setImgTF(const QImage &value)
{
    imgTF     = value;
    videoMode = false;
}

void WorkerTF::setVideoInfo(QString s, QString d, bool sAim, bool sInfTime, QMap<QString,bool> aLabels)
{
    source = s;
    destination = d;
    showAim = sAim;
    showInfTime = sInfTime;
    activeLabels = aLabels;
    videoMode = true;
}

void WorkerTF::work()
{   
    if (!videoMode)
    {
        tf->run(imgTF);
        emit finished();
        emit results(tf->getKindNetwork(),tf->getResults(),tf->getConfidence(),tf->getBoxes(),tf->getMasks(),tf->getInferenceTime());
    }
}

QImage WorkerTF::processImage(QImage img)
{
    // Data
    double        minConf       = tf->getThreshold();
    int           inferenceTime = tf->getInferenceTime();
    QStringList   results       = tf->getResults();
    QList<double> confidence    = tf->getConfidence();
    QList<QRectF> boxes         = tf->getBoxes();
    QList<QImag