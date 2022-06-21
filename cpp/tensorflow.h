#ifndef TENSORFLOW_H
#define TENSORFLOW_H

#include <QObject>
#include <QRectF>
#include <QImage>

#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/kernels/register.h"

using namespace tflite;

class TensorFlow : public QObject
{
    Q_OBJECT
public:
    explicit TensorFlow(QObject *parent = nullptr);
    ~TensorFlow();

    static const int knIMAGE_CLASSIFIER = 1;
    static const int knOBJECT_DETECTION = 2;
    static const int DEF_BOX_DISTANCE   = 10;

signals:

public slots:
    bool init(int imgHeight, int imgWidth);
    bool run(QImage img);
    QString getFilename() const;
    void setFilename(const QString &value);
    QS