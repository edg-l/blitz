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

/// One operand of a non-terminator effectful op, after linearization has chosen
/// which register carries it.
///
/// The single-operand form of [`TermArgs`], for the same reason and with the
/// same discriminant: a `ClassId` is expression identity and has no position,
/// while "which register holds this value here" is a per-block fact. Committing
/// linearization's choice into the CFG leaves one resolution instead of one per
/// consumer.
///
/// `Committed` keeps the class beside the VReg because they answer different
/// questions and neither is derivable from the other: the class is what every
/// pass before linearization reads and what canonicalization rewrites, the VReg
/// is what the schedule carries.
///
/// Like `TermArgs::Committed`, this records what linearization decided and is
/// not the authority past the splitter, which routes an operand through a slot
/// or a reload. The post-split carrier is the barrier instruction's role
/// operands, which every rewrite preserves by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffOperand {
    Class(ClassId),
    Committed { class: ClassId, vreg: VReg },
}

impl EffOperand {
    /// The class, whichever form the operand is in.
    pub fn class(&self) -> ClassId {
        match self {
            EffOperand::Class(cid) => *cid,
            EffOperand::Committed { class, .. } => *class,
        }
    }

    /// The class, mutably, whichever form the operand is in.
    pub fn class_mut(&mut self) -> &mut ClassId {
        match self {
            EffOperand::Class(cid) => cid,
            EffOperand::Committed { class, .. } => class,
        }
    }

    /// The VReg linearization chose, or `None` before the commit.
    ///
    /// Every caller runs after it and could unwrap; returning an option keeps a
    /// pass that is moved before it from reading the absence as "this op has no
    /// operand".
    pub fn vreg(&self) -> Option<VReg> {
        match self {
            EffOperand::Class(_) => None,
            EffOperand::Committed { vreg, .. } => Some(*vreg),
        }
    }

    /// Drop back to `Class`, keeping the class the operand carries.
    ///
    /// For a pass that changes the CFG after the commit and hands it back to
    /// linearization: the class is still what the operand *is*, while the VReg
    /// describes a linearization that is about to be replaced.
    pub fn uncommit(&mut self) {
        *self = EffOperand::Class(self.class());
    }
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

    /// Drop back to `Classes`, keeping the class each argument carries.
    ///
    /// For a pass that changes the CFG after the commit and hands it back to
    /// linearization: the classes are still what the arguments *are*, while the
    /// VRegs describe a linearization that is about to be replaced. Keeping them
    /// would be keeping an answer to a question the next pass re-asks.
    pub fn uncommit(&mut self) {
        if let TermArgs::Committed(v) = self {
            *self = TermArgs::Classes(v.iter().map(|a| a.class).collect());
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
        addr: EffOperand,
        ty: Type,
        result: EffOperand,
    },

    /// Store a value e-class to an address e-class.
    /// `ty` is the type of the value being stored (determines store width).
    Store {
        addr: EffOperand,
        val: EffOperand,
        ty: Type,
    },

    /// Call a named function with the given argument e-classes.
    /// `arg_tys` lists the types of the arguments (determines ABI register assignment).
    /// `ret_tys` lists the types of the return values.
    /// `results` holds the e-graph ClassIds of the `Op::CallResult` nodes that
    /// represent the return values in the pure-op world (one per ret_ty).
    Call {
        func: Symbol,
        args: Vec<EffOperand>,
        arg_tys: Vec<Type>,
        ret_tys: Vec<Type>,
        results: Vec<EffOperand>,
    },

    /// Conditional branch to `bb_true` or `bb_false` depending on flags.
    /// `true_args` / `false_args` are passed as block parameters.
    Branch {
        cond: EffOperand,
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
                f(addr.class());
                f(result.class());
            }
            EffectfulOp::Store { addr, val, .. } => {
                f(addr.class());
                f(val.class());
            }
            EffectfulOp::Call { args, results, .. } => {
                args.iter().map(EffOperand::class).for_each(&mut f);
                results.iter().map(EffOperand::class).for_each(&mut f);
            }
            EffectfulOp::Branch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                f(cond.class());
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
                f(addr.class_mut());
                f(result.class_mut());
            }
            EffectfulOp::Store { addr, val, .. } => {
                f(addr.class_mut());
                f(val.class_mut());
            }
            EffectfulOp::Call { args, results, .. } => {
                args.iter_mut().map(EffOperand::class_mut).for_each(&mut f);
                results
                    .iter_mut()
                    .map(EffOperand::class_mut)
                    .for_each(&mut f);
            }
            EffectfulOp::Branch {
                cond,
                true_args,
                false_args,
                ..
            } => {
                f(cond.class_mut());
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
