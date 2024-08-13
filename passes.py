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
"""Transform passes for the vanilla_accelerator accelerator"""

import tvm
from tvm import tir
from tvm.relay.backend.contrib.uma.api.utils import add_llvm_to_block
#from functools import reduce
import pass_injective
import pass_conv2d
import pass_utils

@tvm.tir.transform.prim_func_pass(opt_level=2)
class VanillaAcceleratorTirPass:
    def transform_function(
        self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        update_func = pass_conv2d.conv2d_pass(func, mod, ctx)
        update_func = pass_injective.add_pass(update_func, mod, ctx)
        return update_func
        #return self._vanilla_accelerator_add_pass(func, mod, ctx)
