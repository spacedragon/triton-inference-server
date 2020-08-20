#!/usr/bin/python

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import os
import re
import sys
import requests as httpreq
from builtins import range
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype

from torchvision import datasets, transforms


FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    if FLAGS.protocol == "http":
        model_name = "identity_model"
        shape = [16]

        # without concurrency
        with client_util.InferenceServerClient(FLAGS.url,
                                               verbose=FLAGS.verbose) as client:
            input_data = (16384 * np.random.randn(*shape)).astype(np.uint32)
            inputs = [
                client_util.InferInput("IN", input_data.shape,
                                       np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)

            outputs = []
            outputs.append(client_util.InferRequestedOutput('OUT'))
            results = client.infer(model_name, inputs, outputs=outputs)
            output_data = results.as_numpy('OUT')
            if output_data is None:
                print("error: expected 'OUT'")
                sys.exit(1)

            if not np.array_equal(output_data, input_data):
                print("error: expected output {} to match input {}".format(
                    output_data, input_datas[i]))
                sys.exit(1)

        request_parallelism = 4
        # with concurrency
        with client_util.InferenceServerClient(FLAGS.url,
                                               concurrency=request_parallelism,
                                               verbose=FLAGS.verbose) as client:
            requests = []
            input_datas = []
            for i in range(request_parallelism):
                input_data = (16384 * np.random.randn(*shape)).astype(np.uint32)
                input_datas.append(input_data)
                inputs = [
                    client_util.InferInput("IN", input_data.shape,
                                           np_to_triton_dtype(input_data.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)

                outputs = []
                outputs.append(client_util.InferRequestedOutput('OUT'))
                requests.append(client.async_infer(model_name, inputs, outputs=outputs))

            for i in range(request_parallelism):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                results = requests[i].get_result()
                print(results)

                output_data = results.as_numpy("OUT")
                if output_data is None:
                    print("error: expected 'OUT'")
                    sys.exit(1)

                if not np.array_equal(output_data, input_datas[i]):
                    print("error: expected output {} to match input {}".format(
                        output_data, input_datas[i]))
                    sys.exit(1)

        model_name = 'pytorch_model'
        with client_util.InferenceServerClient(FLAGS.url,
                                               verbose=FLAGS.verbose) as client:
            input_data = np.ones([1, 1, 28, 28], dtype=np.float32)
            inputs = [
                client_util.InferInput("IN", input_data.shape,
                                       np_to_triton_dtype(input_data.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data)
            outputs = []
            outputs.append(client_util.InferRequestedOutput('OUT'))
            results = client.infer(model_name, inputs, outputs=outputs)
            output_data = results.as_numpy('OUT')
            if output_data is None or len(output_data) != 1:
                print("error: expected 'OUT'")
                sys.exit(1)
            output_test_data = [
                -2.23593, -2.4019134, -2.2534406, -2.234721, -2.4211829,
                -2.2918148, -2.306964, -2.3553405, -2.3035986, -2.241666
            ]

            if not np.allclose(output_data[0], output_test_data):
                print("error: expected output {} to match {}".format(
                    output_data[0], output_test_data))
                sys.exit(1)
