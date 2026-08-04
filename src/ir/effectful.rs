use crate::egraph::extract::VReg;
use crate::ir::condcode::CondCode;
use crate::ir::op::ClassId;
use crate::ir::types::Type;

pub type Symbol = String;
pub type BlockId = u32;

/// One block argument after linearization has chosen its storage.
///
/// Both fields, because they answer different questions and neither is
/// derivable from the other. `vreg` is which register carries the value here;
/// `class` is which expression it is, which lowering asks to tell an argument
/// that *is* the parameter it feeds from one that merely equals it. Nothing
/// resolves `class` to a VReg -- that is what `vreg` already is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermArg {
    pub class: ClassId,
    pub vreg: VReg,
}

/// The values a terminator passes to one successor, as block arguments.
///
/// `Classes` while the IR is still being transformed, `Committed` once
/// linearization has chosen which register carries each value.
///
/// A `ClassId` is expression identity and has no position, while a block
/// argument needs occurrence identity -- which register holds this value, in
/// this block, at this point. One e-class maps to several VRegs and which one is
/// right depends on the block, so a consumer holding only the class has to
/// reconstruct the answer from a position-keyed map. Committing the choice into
/// the CFG the moment linearization makes it leaves one resolution instead of
/// one per consumer.
///
/// One discriminant per argument list, so a half-committed list cannot be built.
///
/// `Committed` records what linearization decided and is not maintained past the
/// splitter: slot routing takes an argument out of the register file entirely,
/// which a list of VRegs cannot express. The post-split carrier is
/// `Op::TerminatorArgs`, whose operands are tagged with argument indices and so
/// can be missing one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TermArgs {
    Classes(Vec<ClassId>),
    Committed(Vec<TermArg>),
}

impl Default for TermArgs {
    fn default() -> Self {
        TermArgs::Classes(Vec::new())
    }
}

impl TermArgs {
    /// An argument list of `ClassId`s, the form every pass before linearization
    /// builds and reads.
    pub fn classes(cids: impl IntoIterator<Item = ClassId>) -> Self {
        TermArgs::Classes(cids.into_iter().collect())
    }

    pub fn len(&self) -> usize {
        match self {
            TermArgs::Classes(v) => v.len(),
            TermArgs::Committed(v) => v.len(),
        }
    }

    /// The `ClassId`s, in argument order, whichever form the list is in.
    pub fn class_ids(&self) -> impl Iterator<Item = ClassId> + '_ {
        let (pre, post) = match self {
            TermArgs::Classes(v) => (Some(v.iter().copied()), None),
            TermArgs::Committed(v) => (None, Some(v.iter().map(|a| a.class))),
        };
        pre.into_iter().flatten().chain(post.into_iter().flatten())
    }

    /// The `ClassId`s, for a caller that runs before the commit by construction.
    ///
    /// # Panics
    /// If the list has been committed.
    pub fn expect_classes(&self) -> &[ClassId] {
        match self {
            TermArgs::Classes(v) => v.as_slice(),
            TermArgs::Committed(_) => {
                panic!("terminator arguments are already committed")
            }
        }
    }

    pub fn expect_classes_mut(&mut self) -> &mut Vec<ClassId> {
        match self {
            TermArgs::Classes(v) => v,
            TermArgs::Committed(_) => {
                panic!("terminator arguments are already committed")
            }
        }
    }

    /// The `ClassId`s, mutably, whichever form the list is in.
    pub fn class_ids_mut(&mut self) -> impl Iterator<Item = &mut ClassId> {
        let (pre, post) = match self {
            TermArgs::Classes(v) => (Some(v.iter_mut()), None),
            TermArgs::Committed(v) => (None, Some(v.iter_mut().map(|a| &mut a.class))),
        };
        pre.into_iter().flatten().chain(post.into_iter().flatten())
    }

    /// The committed arguments, or `None` before the commit.
    ///
    /// Every caller runs after it and could unwrap; returning an option keeps a
    /// pass that is moved before it from reading an empty list as "this edge
    /// passes nothing".
    pub fn as_committed(&self) -> Option<&[TermArg]> {
        match self {
            TermArgs::Classes(_) => None,
            TermArgs::Committed(v) => Some(v.as_slice()),
        }
    }
}

/// Effectful operations that must appear in the CFG skeleton (not the e-graph).
///
/// All operands that are pure computed values are referenced by `ClassId`,
/// pointing into the e-graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectfulOp {
    /// Load a value of type `ty` from the given address e-class.
    /// `result` is the e-graph ClassId of the `Op::LoadResult` node that
    /// represents the loaded value in the pure-op world.
    Load {
        addr: ClassId,
        ty: Type,
        result: ClassId,
    },

    /// Store a value e-class to an address e-class.
    /// `ty` is the type of the value being stored (determines store width).
    Store {
        addr: ClassId,
        val: ClassId,
        ty: Type,
    },

    /// Call a named function with the given argument e-classes.
    /// `arg_tys` lists the types of the arguments (determines ABI register assignment).
    /// `ret_tys` lists the types of the return values.
    /// `results` holds the e-graph ClassIds of the `Op::CallResult` nodes that
    /// represent the return values in the pure-op world (one per ret_ty).
    Call {
        func: Symbol,
        args: Vec<ClassId>,
        arg_tys: Vec<Type>,
        ret_tys: Vec<Type>,
        results: Vec<ClassId>,
    },

    /// Conditional branch to `bb_true` or `bb_false` depending on flags.
    /// `true_args` / `false_args` are passed as block parameters.
    Branch {
        cond: ClassId,
        cc: CondCode,
        bb_true: BlockId,
        bb_false: BlockId,
        true_args: TermArgs,
        false_args: TermArgs,
    },

    /// Unconditional jump to `target` with block arguments.
    Jump { target: BlockId, args: TermArgs },

    /// Return from the function, optionally with a value.
    Ret { val: Option<ClassId> },
}

impl EffectfulOp {
    /// Visit every `ClassId` this op references, in operand order.
    ///
    /// Exhaustive over the enum on purpose: a new variant, or a new `ClassId`
    /// field on an existing one, must fail to compile here rather than be
    /// silently skipped by every walker in the pipeline.
    pub fn for_each_class_id(&self, mut f: impl FnMut(ClassId)) {
        match self {
            EffectfulOp::Load { addr, result, .. } => {
                f(*addr);
                f(*result);
            }
            EffectfulOp::Store { addr, val, .. } => {
                f(*addr);
                f(*val);
            }
            EffectfulOp::Call { args, results, .. } => {
                args.iter().copied().for_each(&mut f);
                results.iter().copied().for_each(&mut f);
            }
            EffectfulOp::Branch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                f(*cond);
                for args in [true_args, false_args] {
                    args.class_ids().for_each(&mut f);
                }
            }
            EffectfulOp::Jump { args, .. } => args.class_ids().for_each(&mut f),
            EffectfulOp::Ret { val } => {
                if let Some(v) = val {
                    f(*v);
                }
            }
        }
    }

    /// Visit every `ClassId` this op references, allowing rewrites.
    ///
    /// Same exhaustiveness rationale as [`Self::for_each_class_id`].
    pub fn for_each_class_id_mut(&mut self, mut f: impl FnMut(&mut ClassId)) {
        match self {
            EffectfulOp::Load { addr, result, .. } => {
                f(addr);
                f(result);
            }
            EffectfulOp::Store { addr, val, .. } => {
                f(addr);
                f(val);
            }
            EffectfulOp::Call { args, results, .. } => {
                args.iter_mut().for_each(&mut f);
                results.iter_mut().for_each(&mut f);
            }
            EffectfulOp::Branch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                f(cond);
                for args in [true_args, false_args] {
                    args.class_ids_mut().for_each(&mut f);
                }
            }
            EffectfulOp::Jump { args, .. } => args.class_ids_mut().for_each(&mut f),
            EffectfulOp::Ret { val } => {
                if let Some(v) = val {
                    f(v);
                }
            }
        }
    }

    /// Returns `true` if this operation is a block terminator.
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            EffectfulOp::Branch { .. } | EffectfulOp::Jump { .. } | EffectfulOp::Ret { .. }
        )
    }

    /// Returns a mutable reference to this op if it is a terminator.
    pub fn as_terminator_mut(&mut self) -> Option<&mut Self> {
        if self.is_terminator() {
            Some(self)
        } else {
            None
        }
    }
}
