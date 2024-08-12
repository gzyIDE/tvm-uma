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
from functools import reduce
import op_injective

def has_block(name: str, func: tvm.tir.PrimFunc) -> bool:
    """
    Determine of a tir.block with `name` exists in `func`
    """
    def _hb(op):
        if isinstance(op, tvm.tir.Block):
            found_blocks.append(op.name_hint)

    found_blocks = []
    tvm.tir.stmt_functor.post_order_visit(func.body, _hb)
    return name in found_blocks

def find_blocks(name: str, func: tvm.tir.PrimFunc) :
    def _hb(op):
        if isinstance(op, tvm.tir.Block):
            found_blocks.append(op.name_hint)

    found_blocks = []
    tvm.tir.stmt_functor.post_order_visit(func.body, _hb)
    return list(filter(lambda x: name in x, found_blocks))

def stmt_analysis(stmt: tvm.tir.Stmt) -> bool:
    def _hb(op):
        if isinstance(op, tvm.tir.Block):
            input_buf.extend(map(lambda x: x.buffer, op.reads))
            output_buf.extend(map(lambda x: x.buffer, op.writes))

    input_buf = []
    output_buf = []
    tvm.tir.stmt_functor.post_order_visit(stmt.body, _hb)
    return (input_buf, output_buf)

@tvm.tir.transform.prim_func_pass(opt_level=2)
class VanillaAcceleratorConv2dPass:
    _EXTERNAL_FUNCTION_NAME = "vanilla_accelerator_conv2dnchw"
    _TVM_BLOCK_MATCH_NAME = "conv2d_nchw"

    def transform_function(
        self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        return self._vanilla_accelerator_conv2d_pass(func, mod, ctx)

    @classmethod
    def _vanilla_accelerator_conv2d_pass(cls, func, mod, ctx):
        _loops = dict()
        _handles = []
        _entry_node = None

        def _detect_and_replace_conv2d(
            func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
        ) -> tvm.tir.PrimFunc:
            def _replace_conv2d(op):
                if op == _entry_node:
                    irb = tvm.tir.ir_builder.create()
                    # Collection of buffer address
                    buffers = [b[1].data for b in _handles]
                    # extraction of loop offsets
                    for k, v in _loops.items():
                        assert v.min.value == 0
                    offset_order = ["co", "w", "h", "ci", "kh", "kw"]
                    offsets = [_loops[i].extent.value for i in offset_order]
                    args = buffers + offsets
                    irb.emit(tir_call(irb, True, cls._EXTERNAL_FUNCTION_NAME, *args))
                    irb_result = irb.get()
                    return irb_result
                elif isinstance(op, tvm.tir.SeqStmt):
                    # Remove that pad block of TOPI's conv2DNCHW by only returning the 2nd statement
                    return op.seq[1]
                return op

            sch = tir.Schedule(func)

            if has_block(cls._TVM_BLOCK_MATCH_NAME, func):
                conv2d_block = sch.get_block(cls._TVM_BLOCK_MATCH_NAME)
                rv_loops = sch.get_loops(conv2d_block)
                assert len(rv_loops) == 7
                loops = dict(
                    n=rv_loops[0],
                    co=rv_loops[1],
                    h=rv_loops[2],
                    w=rv_loops[3],
                    ci=rv_loops[4],
                    kh=rv_loops[5],
                    kw=rv_loops[6],
                )
                _entry_node = sch.get(rv_loops[1])
                _loops = {k: sch.get(v) for k, v in loops.items()}
                _handles = func.buffer_map.items()

                x = tvm.tir.stmt_functor.ir_transform(
                    func.body, None, _replace_conv2d, ["tir.For", "tir.SeqStmt"]
                )
                return func.with_body(x)
            else:
                return func

        r = _detect_and_replace_conv2d(func, mod, ctx)
        return r

@tvm.tir.transform.prim_func_pass(opt_level=2)
class VanillaAcceleratorTirPass:
    _TVM_BLOCK_MATCH_NAME = "T_add"
    _EXTERNAL_FUNCTION_NAME = "vanilla_accelerator_addvec"

    def transform_function(
        self, func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        return self._vanilla_accelerator_add_pass(func, mod, ctx)

    @classmethod
    def _vanilla_accelerator_add_pass(cls, func, mod, ctx):
        _loops = dict()
        _handles = []
        _entry_node = None

        def _detect_and_replace_add(
            func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
        ) -> tvm.tir.PrimFunc:
            def _replace_add(op):

                if op == _entry_node:
                    (inputs, outputs) = stmt_analysis(op)
                    in1       = inputs[0]
                    in2       = inputs[1]
                    out       = outputs[0]
                    in1_elm   = int(reduce(lambda x, y: x * y, in1.shape))
                    in2_elm   = int(reduce(lambda x, y: x * y, in2.shape))
                    if in1_elm < in2_elm :
                        width = in2_elm
                        bc    = in2_elm // in1_elm
                        args  = [in2.data, in1.data, out.data, width, bc]
                        fname = cls._EXTERNAL_FUNCTION_NAME + "_bc"
                    elif in1_elm > in2_elm :
                        width = in1_elm
                        bc    = in1_elm // in2_elm
                        args  = [in1.data, in2.data, out.data, width, bc]
                        fname = cls._EXTERNAL_FUNCTION_NAME + "_bc"
                    else :
                        width = in1_elm
                        args  = [in1.data, in2.data, out.data, width]
                        fname = cls._EXTERNAL_FUNCTION_NAME

                    irb = tvm.tir.ir_builder.create()
                    irb.emit(tir_call(irb, True, fname, *args))
                    return irb.get()
                else:
                    return op

            sch = tir.Schedule(func)

            blk_list = find_blocks(cls._TVM_BLOCK_MATCH_NAME, func)
            if len(blk_list) != 0:
                func_update = func

                for blk in blk_list :
                    add_block   = sch.get_block(blk)
                    rv_loops    = sch.get_loops(add_block)
                    _entry_node = sch.get(rv_loops[0])
                    _loops      = [sch.get(v) for v in rv_loops]
                    _handles    = func_update.buffer_map.items()

                    x = tvm.tir.stmt_functor.ir_transform(
                        func_update.body, None, _replace_add, ["tir.For", "tir.SeqStmt"]
                    )

                    func_update = func.with_body(x)

                return func_update
            else :
                return func


            #if has_block(cls._TVM_BLOCK_MATCH_NAME, func):
            #    add_block   = sch.get_block(cls._TVM_BLOCK_MATCH_NAME)
            #    rv_loops    = sch.get_loops(add_block)
            #    _entry_node = sch.get(rv_loops[0])
            #    _loops      = [sch.get(v) for v in rv_loops]
            #    _handles    = func.buffer_map.items()

            #    x = tvm.tir.stmt_functor.ir_transform(
            #        func.body, None, _replace_add, ["tir.For", "tir.SeqStmt"]
            #    )

            #    return func.with_body(x)
            #else:
            #    return func

        r = _detect_and_replace_add(func, mod, ctx)
        return r


def tir_call(ib: tvm.tir.ir_builder, extern: bool, name: str, *args):
    """
    ib: ir_builder
    extern: bool
        True  --> tvm.tir.call_extern
        False --> tvm.tir.call_packed
    name: str
        function name
    *args:
        arguments for function call
    """

    def buf_from_array(ib, arr, dtype):
        # Allocate enough memory to store the whole array
        var = ib.allocate("int32", (len(arr),), scope="global")
        for i, v in enumerate(arr):
            var[i] = v
        # Declare a buffer, which is basically a view on the chunk of memory that we allocated
        buf = tvm.tir.decl_buffer((len(arr),), dtype, data=var, scope="global")
        return buf

    if extern:
        args = [i.data if isinstance(i, tvm.tir.Buffer) else i for i in args]
        return tvm.tir.call_extern("int32", name, *args)
    else:
        args = [
            buf_from_array(ib, i, "int32")
            if isinstance(i, (tuple, list, tvm.ir.container.Array))
            else i
            for i in args
        ]
        return tvm.tir.call_packed(name, *args)
