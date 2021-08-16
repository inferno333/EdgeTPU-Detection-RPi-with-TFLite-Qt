#include "tensorflow.h"

#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"

#include "edgetpu.h"

#include "get_top_n.h"
#include "colormanager.h"

#include <QDebug>
#include <QString>
#include <QElapsedTimer>
#include <QPointF>
#include <math.h>
#include <auxutils.h>
#include <iostream>
#include <vector>

TensorFlow::TensorFlow(QObject *parent) : QObject(parent)
{
    initialized         = false;
    accelaration        = false;
    verbose             = true;
    numThreads          = 1;
    threshold           = 0.1;
    has_detection_masks = false;
    TPU                 = true;
}

TensorFlow::~TensorFlow(){}

template<class T>
bool formatImageQt(T* out, QImage image, int image_channels, int wanted_height, int wanted_width, int wanted_channels, bool input_floating, bool scale = false)
{
    const float input_mean = 127.5f;
    const float input_std  = 127.5f;

    // Check same number of channels
    if (image_channels != wanted_channels)
    {
        qDebug() << "ERROR: the image has" << image_channels << " channel