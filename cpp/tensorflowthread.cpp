#include "tensorflowthread.h"
#include "auxutils.h"

void WorkerTF::setTf(TensorFlow *value)
{
    tf = value;
}

void WorkerTF::setImgTF(const QImage &value)
{
    imgTF     = value;
    videoMod