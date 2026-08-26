

def test_forced_budget_explore_llm():
    """强制预算(2026-08-26 用户:指定轮全探索+LLM上调):探索偏置+LLM占比覆盖,和为1。"""
    from adaptive import forced_budget, budget_mode
    cfg, reason = forced_budget("explore", 0.4)
    vals = [cfg.mutate, cfg.crossover, cfg.perturb, cfg.random, cfg.llm]
    assert abs(sum(vals) - 1.0) < 1e-6
    assert cfg.llm == 0.4 and cfg.random > 0.15      # 探索:随机与LLM都抬高
    assert cfg.perturb <= 0.05
    assert budget_mode(reason) == "探索"
    # 不带 mode 只调 llm:其它四维等比缩放
    cfg2, _ = forced_budget(None, 0.5, base=cfg)
    assert cfg2.llm == 0.5 and abs(sum([cfg2.mutate, cfg2.crossover, cfg2.perturb,
                                        cfg2.random, cfg2.llm]) - 1.0) < 1e-6
