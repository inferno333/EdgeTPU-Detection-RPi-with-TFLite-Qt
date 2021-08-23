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
        qDebug() << "ERROR: the image has" << image_channels << " channels. Wanted channels:" << wanted_channels;
        return false;
    }

    // Scale image if needed
    if (scale && (image.width() != wanted_width || image.height() != wanted_height))
        image = image.scaled(wanted_height,wanted_width,Qt::IgnoreAspectRatio,Qt::FastTransformation);

    // Number of pixels
    const int numberPixels = image.height()*image.width()*wanted_channels;

    // Pointer to image data
    const uint8_t *output = image.bits();

    // Boolean to [0,1]
    const int inputFloat = input_floating ? 1 : 0;
    const int inputInt   = input_floating ? 0 : 1;

    // Transform to [0,128] ¿?
    for (int i = 0; i < numberPixels; i++)
    {
      out[i] = inputFloat*((output[i] - input_mean) / input_std) + // inputFloat*(output[i]/ 128.f - 1.f) +
               inputInt*(uint8_t)output[i];
      //qDebug() << out[i];
    }

    return true;
}

// -----------------------------------------------------------------------------------------------------------------------
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/examples/label_image/bitmap_helpers_impl.h
// -----------------------------------------------------------------------------------------------------------------------
template <class T>
void formatImageTFLite(T* out, const uint8_t* in, int image_height, int image_width, int image_channels, int wanted_height, int wanted_width, int wanted_channels, bool input_floating)
{
   const float input_mean = 127.5f;
   const float input_std  = 127.5f;

  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<Interpreter> interpreter(new Interpreter);

  int base_index = 0;

  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);

  // one output
  interpreter->AddTensors(1, &base_index);

  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});

  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input",    {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32,   "new_size", {2},quant);
  interpreter->SetTensorParametersReadWrite(2, kTfLiteFloat32, "output",   {1, wanted_height, wanted_width, wanted_channels}, quant);

  ops::builtin::BuiltinOpResolver resolver;
  const TfLiteRegistration *resize_op = resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR,1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(malloc(size