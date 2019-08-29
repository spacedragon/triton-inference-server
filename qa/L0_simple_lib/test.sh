#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

SIMPLE_CLIENT=./simple
CLIENT_LOG="./client.log"
MODELSDIR=`pwd`/models

DATADIR=/data/inferenceserver/qa_model_repository

export CUDA_VISIBLE_DEVICES=0

rm -f $CLIENT_LOG

RET=0

# Apply the same procedure to addsub models in other frameworks
for trial in \
        graphdef_float32_float32_float32 \
        savedmodel_float32_float32_float32 \
        netdef_float32_float32_float32 \
        onnx_float32_float32_float32 \
        libtorch_float32_float32_float32 \
        plan_float32_float32_float32 ; do
    rm -rf $MODELSDIR/simple
    mkdir -p $MODELSDIR/simple/1 && \
        cp -r $DATADIR/${trial}/1/* $MODELSDIR/simple/1/. && \
        cp $DATADIR/${trial}/config.pbtxt $MODELSDIR/simple/. && \
        (cd $MODELSDIR/simple && \
                sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt && \
                sed -i "s/label_filename:.*//" config.pbtxt)

    set +e

    $SIMPLE_CLIENT -r $MODELSDIR >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $SIMPLE_CLIENT -r $MODELSDIR -g >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    set -e
done

rm -rf $MODELSDIR/simple/1/*
cp -r ../custom_models/custom_float32_float32_float32/1/* $MODELSDIR/simple/1/.
cp ../custom_models/custom_float32_float32_float32/config.pbtxt $MODELSDIR/simple/.
(cd $MODELSDIR/simple && \
            sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt && \
            sed -i "s/label_filename:.*//" config.pbtxt)

set +e

$SIMPLE_CLIENT -r $MODELSDIR >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$SIMPLE_CLIENT -r $MODELSDIR -g >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
