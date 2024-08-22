import tvm
from tvm import tir
import pass_utils

def add_device_buffer(func, mod, ctx):
    sch = tir.Schedule(func)

    def _hb(op):
        print(type(op))

    tvm.tir.stmt_functor.post_order_visit(func.body, _hb)

    #print(func)
    #block = sch.get_block("root", func_name = "tvmgen_default_vanilla_accelerator_main_0")
    #block = sch.get_block("main")
    #block = sch.get_block("T_add")
    #print(sch.mod.show())
    ##i, j, k, l = sch.get_loops("T_add")
    ##print(type(block))
    #sch.cache_read(block, 0, "local")
    #sch.cache_read(block, 1, "local")
    #sch.cache_write(block, 0, "local")
    print(sch.mod.show())

    return func
