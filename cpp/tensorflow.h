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
    QString getLabelsFilename() const;
    void setLabelsFilename(const QString &value);
    bool getAccelaration() const;
    void setAccelaration(bool value);
    bool getVerbose() const;
    void setVerbose(bool value);
    int getNumThreads() const;
    void setNumThreads(int value);
    int getHeight() const;
    int getWidth() const;
    int getChannels() const;
    QString getLabel(int index);
    QString getResultCaption(int index);
    double  getResultConfidence(int index);
    QStringList getResults();
    QList<double> getConfidence();
    QList<QRectF> getBoxes();
    QList<QImage> getMasks();
    int getInferenceTime();
    int getKindNetwork();
    double getThreshold() const;
    void setThreshold(double value);
    void initInput(int imgHeight, int imgWidth);
    bool initTFLite(int imgHeight, int imgWidth);
    bool setInputsTFLite(QImage image);
    bool 