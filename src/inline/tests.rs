#[cfg(test)]
mod tests {
    use crate::compile::CompileOptions;
    use crate::inline::callgraph::{build_call_graph, is_recursive, should_inline};
    use crate::inline::inline_module;
    use crate::inline::transform::inline_call_site;
    use crate::ir::builder::FunctionBuilder;
    use crate::ir::effectful::EffectfulOp;
    use crate::ir::function::Function;
    use crate::ir::op::Op;
    use crate::ir::types::Type;

    fn inline_opts() -> CompileOptions {
        CompileOptions {
            enable_inlining: true,
            max_inline_depth: 3,
            max_inline_nodes: 50,
            ..Default::default()
        }
    }

    /// Build a simple leaf callee: fn leaf() -> i64 { return 42; }
    fn build_leaf() -> Function {
        let mut b = FunctionBuilder::new("leaf", &[], &[Type::I64]);
        let c = b.iconst(42, Type::I64);
        b.ret(Some(c));
        b.finalize().expect("leaf finalize")
    }

    /// Build a void callee: fn void_fn() { return; }
    fn build_void() -> Function {
        let mut b = FunctionBuilder::new("void_fn", &[], &[]);
        b.ret(None);
        b.finalize().expect("void finalize")
    }

    /// Build a callee with stack slots: fn with_slots(x: i64) -> i64 { ... }
    fn build_with_slots() -> Function {
        let mut b = FunctionBuilder::new("with_slots", &[Type::I64], &[Type::I64]);
        let slot1 = b.create_stack_slot(8, 8);
        let _slot2 = b.create_stack_slot(16, 8);
        let addr = b.stack_addr(slot1);
        let params = b.params().to_vec();
        b.store(addr, params[0]);
        let loaded = b.load(addr, Type::I64);
        b.ret(Some(loaded));
        b.finalize().expect("with_slots finalize")
    }

    /// Build a caller that calls leaf(): fn caller() -> i64 { return leaf(); }
    fn build_caller_of_leaf() -> Function {
        let mut b = FunctionBuilder::new("main", &[], &[Type::I64]);
        let results = b.call("leaf", &[], &[Type::I64]);
        b.ret(Some(results[0]));
        b.finalize().expect("caller finalize")
    }

    /// Build a caller that calls void_fn(): fn caller() { void_fn(); return; }
    fn build_caller_of_void() -> Function {
        let mut b = FunctionBuilder::new("main", &[], &[]);
        b.call("void_fn", &[], &[]);
        b.ret(None);
        b.finalize().expect("caller finalize")
    }

    /// Build a caller that calls with_slots(10): fn caller() -> i64 { return with_slots(10); }
    fn build_caller_of_slots() -> Function {
        let mut b = FunctionBuilder::new("main", &[], &[Type::I64]);
        let arg = b.iconst(10, Type::I64);
        let results = b.call("with_slots", &[arg], &[Type::I64]);
        b.ret(Some(results[0]));
        b.finalize().expect("caller finalize")
    }

    // ── Call graph tests ─────────────────────────────────────────────────────

    #[test]
    fn test_callgraph_build() {
        let caller = build_caller_of_leaf();
        let leaf = build_leaf();
        let void_fn = build_void();

        let functions = vec![caller, leaf, void_fn];
        let graph = build_call_graph(&functions);

        assert!(graph["main"].contains("leaf"));
        assert!(!graph["main"].contains("void_fn"));
        assert!(graph["leaf"].is_empty());
        assert!(graph["void_fn"].is_empty());
    }

    #[test]
    fn test_recursive_detection_direct() {
        // Build a function that calls itself.
        let mut b = FunctionBuilder::new("rec", &[Type::I64], &[Type::I64]);
        let params = b.params().to_vec();
        let results = b.call("rec", &[params[0]], &[Type::I64]);
        b.ret(Some(results[0]));
        let rec_fn = b.finalize().expect("rec finalize");

        let graph = build_call_graph(&[rec_fn]);
        assert!(is_recursive("rec", &graph));
    }

    #[test]
    fn test_recursive_detection_mutual() {
        // a calls b, b calls a.
        let mut ba = FunctionBuilder::new("a", &[Type::I64], &[Type::I64]);
        let pa = ba.params().to_vec();
        let ra = ba.call("b", &[pa[0]], &[Type::I64]);
        ba.ret(Some(ra[0]));
        let fn_a = ba.finalize().unwrap();

        let mut bb = FunctionBuilder::new("b", &[Type::I64], &[Type::I64]);
        let pb = bb.params().to_vec();
        let rb = bb.call("a", &[pb[0]], &[Type::I64]);
        bb.ret(Some(rb[0]));
        let fn_b = bb.finalize().unwrap();

        let graph = build_call_graph(&[fn_a, fn_b]);
        assert!(is_recursive("a", &graph));
        assert!(is_recursive("b", &graph));
    }

    #[test]
    fn test_non_recursive_not_detected() {
        let caller = build_caller_of_leaf();
        let leaf = build_leaf();
        let graph = build_call_graph(&[caller, leaf]);
        assert!(!is_recursive("main", &graph));
        assert!(!is_recursive("leaf", &graph));
    }

    // ── should_inline tests ──────────────────────────────────────────────────

    #[test]
    fn test_should_inline_simple_leaf() {
        let opts = inline_opts();
        let leaf = build_leaf();
        assert!(should_inline(&leaf, 0, &opts));
    }

    #[test]
    fn test_inline_multi_return_skipped() {
        let opts = inline_opts();
        let mut f = Function::new("multi_ret", vec![Type::I64], vec![Type::I64, Type::I32]);
        // Multi-return functions should not be inlined.
        assert!(!should_inline(&f, 0, &opts));
        let _ = &mut f;
    }

    #[test]
    fn test_inline_depth_limit() {
        let opts = inline_opts();
        let leaf = build_leaf();
        // At depth >= max_inline_depth, should return false.
        assert!(!should_inline(&leaf, 3, &opts));
        assert!(should_inline(&leaf, 2, &opts));
    }

    #[test]
    fn test_inline_no_egraph_skipped() {
        let opts = inline_opts();
        let f = Function::new("no_eg", vec![], vec![Type::I64]);
        assert!(!should_inline(&f, 0, &opts));
    }

    // ── Inline transform tests ───────────────────────────────────────────────

    #[test]
    fn test_inline_simple_leaf() {
        let mut caller = build_caller_of_leaf();
        let leaf = build_leaf();

        // Before: caller has a Call op.
        let has_call_before = caller.blocks.iter().any(|b| {
            b.ops
                .iter()
                .any(|op| matches!(op, EffectfulOp::Call { .. }))
        });
        assert!(has_call_before);

        inline_call_site(&mut caller, 0, 0, &leaf);

        // After: no Call ops remain.
        let has_call_after = caller.blocks.iter().any(|b| {
            b.ops
                .iter()
                .any(|op| matches!(op, EffectfulOp::Call { .. }))
        });
        assert!(!has_call_after);

        // The callee's Iconst(42) should have been imported into the caller's egraph.
        let egraph = caller.egraph.as_ref().unwrap();
        let has_42 = egraph.classes.iter().any(|c| {
            c.nodes
                .iter()
                .any(|n| matches!(n.op, Op::Iconst(42, Type::I64)))
        });
        assert!(
            has_42,
            "Iconst(42) should be in caller egraph after inlining"
        );
    }

    #[test]
    fn test_inline_void_callee() {
        let mut caller = build_caller_of_void();
        let void_fn = build_void();

        inline_call_site(&mut caller, 0, 0, &void_fn);

        // No Call ops remain.
        let has_call = caller.blocks.iter().any(|b| {
            b.ops
                .iter()
                .any(|op| matches!(op, EffectfulOp::Call { .. }))
        });
        assert!(!has_call);

        // Find the continuation block. It should have no block params (void return).
        let cont_block = caller.blocks.last().unwrap();
        assert!(
            cont_block.param_types.is_empty(),
            "void callee should produce continuation block with no params"
        );
    }

    #[test]
    fn test_inline_with_stack_slots() {
        let mut caller = build_caller_of_slots();
        let callee = build_with_slots();

        let caller_slots_before = caller.stack_slots.len();
        let callee_slots = callee.stack_slots.len();

        inline_call_site(&mut caller, 0, 0, &callee);

        // Caller should have gained the callee's stack slots.
        assert_eq!(caller.stack_slots.len(), caller_slots_before + callee_slots);

        // Verify StackAddr ops in the egraph have been remapped (offset by caller_slots_before).
        let egraph = caller.egraph.as_ref().unwrap();
        let remapped_stack_addrs: Vec<u32> = egraph
            .classes
            .iter()
            .flat_map(|c| {
                c.nodes.iter().filter_map(|n| {
                    if let Op::StackAddr(slot) = n.op {
                        if slot >= caller_slots_before as u32 {
                            return Some(slot);
                        }
                    }
                    None
                })
            })
            .collect();
        assert!(
            !remapped_stack_addrs.is_empty(),
            "should have remapped StackAddr ops"
        );
    }

    #[test]
    fn test_inline_recursive_skipped_in_module() {
        let mut b = FunctionBuilder::new("main", &[], &[Type::I64]);
        let r = b.call("rec", &[], &[Type::I64]);
        b.ret(Some(r[0]));
        let main_fn = b.finalize().unwrap();

        let mut br = FunctionBuilder::new("rec", &[], &[Type::I64]);
        let rr = br.call("rec", &[], &[Type::I64]);
        br.ret(Some(rr[0]));
        let rec_fn = br.finalize().unwrap();

        let mut functions = vec![main_fn, rec_fn];
        let opts = inline_opts();
        inline_module(&mut functions, &opts);

        // rec should not have been inlined into main (it's recursive).
        let main = &functions[0];
        let has_call = main.blocks.iter().any(|b| {
            b.ops
                .iter()
                .any(|op| matches!(op, EffectfulOp::Call { .. }))
        });
        assert!(has_call, "recursive function should not be inlined");
    }

    // ── Module-level inline + dead function elimination ──────────────────────

    #[test]
    fn test_inline_module_eliminates_dead() {
        let caller = build_caller_of_leaf();
        let leaf = build_leaf();

        let mut functions = vec![caller, leaf];
        let opts = inline_opts();
        inline_module(&mut functions, &opts);

        // After inlining leaf into main and dead function elimination,
        // leaf should be removed (no remaining call sites).
        let names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"main"), "main should be kept");
        assert!(!names.contains(&"leaf"), "leaf should be eliminated");
    }

    #[test]
    fn test_inline_module_keeps_called() {
        // Build a scenario where a function is called but not inlinable (too large).
        let caller = build_caller_of_leaf();
        let leaf = build_leaf();

        let mut functions = vec![caller, leaf];
        let opts = CompileOptions {
            enable_inlining: true,
            max_inline_nodes: 0, // too small, nothing gets inlined
            ..Default::default()
        };
        inline_module(&mut functions, &opts);

        // leaf should still be present since it wasn't inlined.
        let names: Vec<&str> = functions.iter().map(|f| f.name.as_str()).collect();
        assert!(
            names.contains(&"leaf"),
            "called but not inlined function should be kept"
        );
    }
}
