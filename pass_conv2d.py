import tvm
from tvm import tir
from functools import reduce
import pass_utils

def conv2d_pass(func, mod, ctx):
    _loops = dict()
    _entry_node = None

    def _detect_and_replace_pad(
        func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        def _replace_pad(op):
            if op == _entry_node:
                (inputs, outputs) = pass_utils.stmt_analysis(op)

                irb = tvm.tir.ir_builder.create()
                # extraction of loop offsets
                for k, v in _loops.items():
                    assert v.min.value == 0
                offset_order = ["w", "h", "ci"]
                offsets = [_loops[i].extent.value for i in offset_order]
                args = inputs + outputs + offsets + [1, 1]
                irb.emit(pass_utils.tir_call(irb, True, "vanilla_accelerator_pad", *args))
                irb_result = irb.get()
                return irb_result
            else:
                return op

        sch = tir.Schedule(func)

        if pass_utils.has_block("pad_temp", func):
            pad_block = sch.get_block("pad_temp")
            rv_loops = sch.get_loops(pad_block)
            assert len(rv_loops) == 4
            loops = dict(
                n  = rv_loops[0],
                ci = rv_loops[1],
                h  = rv_loops[2],
                w  = rv_loops[3],
            )
            _entry_node = sch.get(rv_loops[1])
            _loops = {k: sch.get(v) for k, v in loops.items()}

            x = tvm.tir.stmt_functor.ir_transform(
                func.body, None, _replace_pad, ["tir.For", "tir.SeqStmt"]
            )
            return func.with_body(x)
        else:
            return func

    def _detect_and_replace_conv2d(
        func: tvm.tir.PrimFunc, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.tir.PrimFunc:
        def _replace_conv2d(op):
            if op == _entry_node:
                (inputs, outputs) = pass_utils.stmt_analysis(op)

                irb = tvm.tir.ir_builder.create()
                # extraction of loop offsets
                for k, v in _loops.items():
                    assert v.min.value == 0
                offset_order = ["co", "w", "h", "ci", "kh", "kw"]
                offsets = [_loops[i].extent.value for i in offset_order]
                args = inputs + outputs + offsets
                irb.emit(pass_utils.tir_call(irb, True, "vanilla_accelerator_conv2dnchw", *args))
                irb_result = irb.get()
                return irb_result
            else:
                return op

        sch = tir.Schedule(func)

        if pass_utils.has_block("conv2d_nchw", func):
            conv2d_block = sch.get_block("conv2d_nchw")
            rv_loops = sch.get_loops(conv2d_block)
            assert len(rv_loops) == 7
            loops = dict(
                n  =rv_loops[0],
                co =rv_loops[1],
                h  =rv_loops[2],
                w  =rv_loops[3],
                ci =rv_loops[4],
                kh =rv_loops[5],
                kw =rv_loops[6],
            )
            _entry_node = sch.get(rv_loops[1])
            _loops = {k: sch.get(v) for k, v in loops.items()}

            x = tvm.tir.stmt_functor.ir_transform(
                func.body, None, _replace_conv2d, ["tir.For", "tir.SeqStmt"]
            )
            return func.with_body(x)
        else:
            return func

    r = _detect_and_replace_pad(func, mod, ctx)
    r = _detect_and_replace_conv2d(r, mod, ctx)
    return r
