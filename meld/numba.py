import numba as nb
from numba.core import bytecode

from numba.core import (utils, errors, typing, interpreter, bytecode, postproc,
                        config, callconv, cpu)

from numba.core.tracing import event

class PartialCompiler(nb.core.compiler.Compiler):

    def compile_partial(self, func):
        """
        Populate and run compiler pipeline, but avoid lowering passes
        """
        # Code to initialize the compiler
        self.state.func_id = bytecode.FunctionIdentity.from_function(func)
        nb.core.untyped_passes.ExtractByteCode().run_pass(self.state)

        self.state.lifted = ()
        self.state.lifted_from = None

        with nb.core.targetconfig.ConfigStack().enter(self.state.flags.copy()):
            pms = self.define_pipelines()
            for pm in pms:
                if isinstance(pm, nb.core.compiler_machinery.LoweringPass):
                    # Skip lowering passes
                    continue
                pipeline_name = pm.pipeline_name
                func_name = "%s.%s" % (self.state.func_id.modname,
                                       self.state.func_id.func_qualname)

                event("Pipeline: %s for %s" % (pipeline_name, func_name))
                self.state.metadata['pipeline_times'] = {pipeline_name:
                                                         pm.exec_times}
                is_final_pipeline = pm == pms[-1]
                res = None
                try:
                    pm.run(self.state)
                    if self.state.cr is not None:
                        break
                except _EarlyPipelineCompletion as e:
                    res = e.result
                    break
                except Exception as e:
                    if (utils.use_new_style_errors() and not
                            isinstance(e, errors.NumbaError)):
                        raise e

                    self.state.status.fail_reason = e
                    if is_final_pipeline:
                        raise e
            else:
                raise nb.core.errors.CompilerError("All available pipelines exhausted")

            # Pipeline is done, remove self reference to release refs to user
            # code
            self.state.pipeline = None

            # organise a return
            if res is not None:
                # Early pipeline completion
                return res
            else:
                return self.state.func_ir

    def compile_final(self):
        """
        Finish compile pipeline after building func IR
        """
        with ConfigStack().enter(self.state.flags.copy()):
            pms = self.define_pipelines()
            for pm in pms:
                if not isinstance(pm, nb.core.compiler_machinery.LoweringPass):
                    # Only perform lowering passes
                    continue
                pipeline_name = pm.pipeline_name
                func_name = "%s.%s" % (self.state.func_id.modname,
                                       self.state.func_id.func_qualname)

                event("Pipeline: %s for %s" % (pipeline_name, func_name))
                self.state.metadata['pipeline_times'] = {pipeline_name:
                                                         pm.exec_times}
                is_final_pipeline = pm == pms[-1]
                res = None
                try:
                    pm.run(self.state)
                    if self.state.cr is not None:
                        break
                except nb.core.compiler._EarlyPipelineCompletion as e:
                    res = e.result
                    break
                except Exception as e:
                    if (utils.use_new_style_errors() and not
                            isinstance(e, errors.NumbaError)):
                        raise e

                    self.state.status.fail_reason = e
                    if is_final_pipeline:
                        raise e
            else:
                raise CompilerError("All available pipelines exhausted")

            # Pipeline is done, remove self reference to release refs to user
            # code
            self.state.pipeline = None

            # organise a return
            if res is not None:
                # Early pipeline completion
                return res
            else:
                assert self.state.cr is not None
                return self.state.cr

def numba_ir_test(func, args, return_type=None, flags=nb.core.compiler.DEFAULT_FLAGS,
                  locals={}):
    """
    Compile the function in an isolated environment (typing and target
    context).
    Good for testing.
    """
    from numba.core.registry import cpu_target
    typingctx = nb.core.typing.Context()
    targetctx = nb.core.cpu.CPUContext(typingctx, target='cpu')
    # Register the contexts in case for nested @jit or @overload calls
    #with cpu_target.nested_context(typingctx, targetctx):
    #    return compile_extra(typingctx, targetctx, func, args, return_type,
    #                         flags, locals)

    library = None
    pipeline = PartialCompiler(typingctx, targetctx, library,
                               args, return_type, flags, locals)

    return pipeline.compile_partial(func), pipeline


def numba_canonical_ir(func):
    # Get the first IR Numba builds from a function
    return nb.core.compiler.run_frontend(func)
