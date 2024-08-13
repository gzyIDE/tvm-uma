import tvm

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
