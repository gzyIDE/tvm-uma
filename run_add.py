# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from backend import VanillaAcceleratorBackend
from tvm.relay import transform
from collections import OrderedDict
import numpy as np


from tvm.testing.aot import (
    AOTTestModel as AOTModel,
    AOTTestRunner as AOTRunner,
    generate_ref_data,
    compile_and_run,
)

#def create():
#    dtype       = "float32"
#    ishape      = (1, 16, 14, 14)
#    runner      = AOT_DEFAULT_RUNNER
#    data0       = relay.var("data0", shape=ishape, dtype=dtype)
#    data1       = relay.var("data1", shape=ishape, dtype=dtype)
#    out         = relay.add(data0, data1)
#    main_f      = relay.Function([data0, data1], out)
#    mod         = tvm.IRModule()
#    mod["main"] = main_f
#    mod         = transform.InferType()(mod)
#    i_data0     = np.random.uniform(0, 1, ishape).astype(dtype)
#    i_data1     = np.random.uniform(0, 1, ishape).astype(dtype)
#    inputs      = OrderedDict([("data0", i_data0), ("data1", i_data1)])
#    output_list = generate_ref_data(mod, inputs)
#    return mod, inputs, output_list, runner

def create():
    dtype       = "float32"
    ishape1     = (1, 16, 14, 14)
    ishape2     = (1, 16, 1, 1)
    runner      = AOT_DEFAULT_RUNNER
    data0       = relay.var("data0", shape=ishape1, dtype=dtype)
    data1       = relay.var("data1", shape=ishape2, dtype=dtype)
    out         = relay.add(data0, data1)
    main_f      = relay.Function([data0, data1], out)
    mod         = tvm.IRModule()
    mod["main"] = main_f
    mod         = transform.InferType()(mod)
    i_data0     = np.random.uniform(0, 1, ishape1).astype(dtype)
    i_data1     = np.random.uniform(0, 1, ishape2).astype(dtype)
    inputs      = OrderedDict([("data0", i_data0), ("data1", i_data1)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, runner


def main():
    mod, inputs, output_list, runner = create()

    #with open("model_pre.dump", "w") as f:
    #    f.write(str(mod))

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    #with open("model_post.dump", "w") as f:
    #    f.write(str(mod))

    #export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path
    export_directory = "result"
    print(f"Generated files are in {export_directory}")
    compile_and_run(
        AOTModel(module=mod, inputs=inputs, outputs=output_list),
        runner,
        interface_api="c",
        use_unpacked_api=True,
        target=[target_c, target],
        test_dir=str(export_directory),
    )


if __name__ == "__main__":
    main()
