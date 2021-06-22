/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//
// This header file defines EdgeTpuManager, and EdgeTpuContext.
// EdgeTpuContext is an object associated with one or more tflite::Interpreter.
// Instances of this class should be allocated through
// EdgeTpuManager::NewEdgeTpuContext.
// More than one Interpreter instances can point to the same context. This means
// the tasks from both would be executed under the same TPU context.
// The lifetime of this context must be longer than all associated
// tflite::Interpreter instances.
//
// Typical usage with NNAPI:
//
//   std::unique_ptr<tflite::Interpreter> interpreter;
//   tflite::ops::builtin::BuiltinOpResolver resolver;
//   auto model =
//   tflite::FlatBufferModel::BuildFromFile(model_file_name.c_str());
//
//   // Registers edge TPU custom op handler with Tflite resolver.
//   resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
//
//   tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//
//   interpreter->AllocateTensors();
//      .... (Prepare input tensors)
//   interpreter->Invoke();
//      .... (retrieving the result from output tensors)
//
//   // Releases interpreter instance to free up resources associated with
//   // this custom op.
//   interpreter.reset();
//
// Typical usage with Non-NNAPI:
//
//   // Sets up the tpu_context.
//   auto tpu_context =
//       edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext();
//
//   std::unique_ptr<tflite::Interpreter> interpreter;
//   tflite::ops::builtin::BuiltinOpResolver resolver;
//   auto model =
//   tflite::FlatBufferModel::BuildFromFile(model_file_name.c_str());
//
//   // Registers edge TPU custom op handler with Tflite resolver.
//   resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
//
//   tflite::InterpreterBuilder(*model, resolver)(&interpreter);
//
//   // Binds a context with a specific interpreter.
//   interpreter->SetExternalContext(kTfLiteEdgeTpuContext,
//     tpu_context.get());
//
//   // Note that all edge TPU context set ups should be done before this
//   // function is called.
//   interpreter-