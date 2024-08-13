import tvm
from tvm import tir
from functools import reduce
import pass_utils

tir_func = {"T_add": "vanilla_accelerator_addvec"}

def add_pass(func, mod, ctx):
    #_loops = dict()
    #_handles = []
    _entry_node = None

    def _detect_and_replace_add(
        func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        def _replace_add(op):
            if op == _entry_node:
                (inputs, outputs) = pass_utils.stmt_analysis(op)
                in1       = inputs[0]
                in2       = inputs[1]
                out       = outputs[0]
                in1_elm   = int(reduce(lambda x, y: x * y, in1.shape))
                in2_elm   = int(reduce(lambda x, y: x * y, in2.shape))
                if in1_elm < in2_elm :
                    width = in2_elm
                    bc    = in2_elm // in1_elm
                    args  = [in2.data, in1.data, out.data, width, bc]
                    fname = tir_func["T_add"] + "_bc"
                elif in1_elm > in2_elm :
                    width = in1_elm
                    bc    = in1_elm // in2_elm
                    args  = [in1.data, in2.data, out.data, width, bc]
                    fname = tir_func["T_add"] + "_bc"
                else :
                    width = in1_elm
                    args  = [in1.data, in2.data, out.data, width]
                    fname = tir_func["T_add"]

                irb = tvm.tir.ir_builder.create()
                irb.emit(pass_utils.tir_call(irb, True, fname, *args))
                return irb.get()
            else:
                return op

        sch = tir.Schedule(func)

        blk_list = pass_utils.find_blocks("T_add", func)
        if len(blk_list) != 0:
            func_update = func

            for blk in blk_list :
                add_block   = sch.get_block(blk)
                rv_loops    = sch.get_loops(add_block)
                _entry_node = sch.get(rv_loops[0])
                #_loops      = [sch.get(v) for v in rv_loops]
                #_handles    = func_update.buffer_map.items()

                x = tvm.tir.stmt_functor.ir_transform(
                    func_update.body, None, _replace_add, ["tir.For", "tir.SeqStmt"]
                )

                func_update = func.with_body(x)

            return func_update
        else :
            return func

    r = _detect_and_replace_add(func, mod, ctx)
    return r
