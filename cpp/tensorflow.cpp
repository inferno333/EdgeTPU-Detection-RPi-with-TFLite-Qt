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
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op, nullptr);
  interpreter->AllocateTensors();


  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++)
    input[i] = in[i];

  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;

  interpreter->Invoke();

  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels = wanted_height * wanted_height * wanted_channels;

  for (int i = 0; i < output_number_of_pixels; i++)
  {
    if (input_floating)
      out[i] = (output[i] - input_mean) / input_std;
    else
      out[i] = (uint8_t)output[i];
  }
}
bool TensorFlow::init(int imgHeight, int imgWidth)
{
    if (!initialized)
        initialized = initTFLite(imgHeight,imgWidth);

    return initialized;
}

void TensorFlow::initInput(int imgHeight, int imgWidth)
{
     Q_UNUSED(imgHeight);
     Q_UNUSED(imgWidth);
}

// ------------------------------------------------------------------------------------------------------------------------------
// Adapted from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/examples/label_image/label_image.cc
// Adapted for TPU edge model from: https://coral.withgoogle.com/docs/edgetpu/api-cpp/
// ------------------------------------------------------------------------------------------------------------------------------
bool TensorFlow::initTFLite(int imgHeight, int imgWidth)
{
    Q_UNUSED(imgHeight);
    Q_UNUSED(imgWidth);

    try{
        // Open model & assign error reporter
        model = AuxUtils::getDefaultModelFilename().trimmed().isEmpty() && AuxUtils::getDefaultLabelsFilename().trimmed().isEmpty() ? nullptr :
                FlatBufferModel::BuildFromFile(filename.toStdString().c_str(),&error_reporter);

        edgetpu::EdgeTpuContext* edgetpu_context;

        if(model == nullptr)
        {
            qDebug() << "TensorFlow model loading: ERROR";
            return false;
        }

        // TPU support
        if (getTPU())
        {
            edgetpu::EdgeTpuManager *edgetpu_manager = edgetpu::EdgeTpuManager::GetSingleton();

            if (edgetpu_manager == nullptr)
            {
                qDebug() << "TPU unsupported on the current platform";
                return false;
            }

            //if (verbose)
            //{
            //    const auto& available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
            //    qDebug() << "Number of TPUs: " << available_tpus.size();
            //    for(auto edgetpu : available_tpus)
            //        qDebug() << "TPU:" << (edgetpu.type == edgetpu::DeviceType::kApexUsb ? "USB" : "PCI") << edgetpu.path.c_str();
            //    qDebug() << "EdgeTPU runtime stack version: " << edgetpu_manager->Version().c_str();
            //}

            edgetpu_context = edgetpu_manager->NewEdgeTpuContext().release();

            if (edgetpu_context == nullptr)
            {
                qDebug() << "TPU cannot be found or opened!";
                return false;
            }

            edgetpu_manager->SetVerbosity(0);
            resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
        }

        // Link model & resolver
        InterpreterBuilder builder(*model.get(), resolver);

        // Check interpreter
        if(builder(&interpreter) != kTfLiteOk)
        {
            qDebug() << "Interpreter: ERROR";
            return false;
        }

        // TPU context
        if (getTPU())
            interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);

        // Apply accelaration (Neural Network Android)
        interpreter->UseNNAPI(accelaration);

        if(interpreter->AllocateTensors() != kTfLiteOk)
        {
            qDebug() << "Allocate tensors: ERROR";
            return false;
        }

        // Set kind of network
        kind_network = interpreter->outputs().size()>1 ? knOBJECT_DETECTION : knIMAGE_CLASSIFIER;

        if (verbose)
        {
          int i_size = interpreter->inputs().size();
          int o_size = interpreter->outputs().size();
          int t_size = interpreter->tensors_size();

          qDebug() << "tensors size: "  << t_size;
          qDebug() << "nodes size: "    << interpreter->nodes_size();
          qDebug() << "inputs: "        << i_size;
          qDebug() << "outputs: "       << o_size;

          for (int i = 0; i < i_size; i++)
            qDebug() << "input" << i << "name:" << interpreter->GetInputName(i) << ", type:" << interpreter->tensor(interpreter->inputs()[i])->type;

          for (int i = 0; i < o_size; i++)
            qDebug() << "output" << i << "name:" << interpreter->GetOutputName(i) << ", type:" << interpreter->tensor(interpreter->outputs()[i])->type;

//          for (int i = 0; i < t_size; i++)
//          {
//            if (interpreter->tensor(i)->name)
//              qDebug()  << i << ":" << interpreter->tensor(i)->name << ","
//                        << interpreter->tensor(i)->bytes << ","
//                        << interpreter->tensor(i)->type << ","
//                        << interpreter->tensor(i)->params.scale << ","
//                        << interpreter->tensor(i)->params.zero_point;
//          }
        }

        // Get input dimension from the input tensor metadata
        // Assuming one input only
        int input = interpreter->inputs()[0];
        TfLiteIntArray* dims = interpreter->tensor(input)->dims;

        // Save outputs
        outputs.clear();
        for(unsigned int i=0;i<interpreter->outputs().size();i++)
            outputs.push_back(interpreter->tensor(interpreter->outputs()[i]));

        wanted_height   = dims->data[1];
        wanted_width    = dims->data[2];
        wanted_channels = dims->data[3];

        if (verbose)
        {
            qDebug() << "Wanted height:"   << wanted_height;
            qDebug() << "Wanted width:"    << wanted_width;
            qDebug() << "Wanted channels:" << wanted_channels;
        }

        if (numThreads > 1)
          interpreter->SetNumThreads(numThreads);

        // Read labels
        if (readLabels()) qDebug() << "There are" << labels.count() << "labels.";
        else qDebug() << "There are NO labels";

        qDebug() << "Tensorflow initialization: OK";
        return true;

    }catch(...)
    {
        qDebug() << "Exception loading model";
        return false;
    }
}

// --------------------------------------------------------------------------