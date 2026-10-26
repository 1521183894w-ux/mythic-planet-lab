"""U35-C9：B5实际侧向通量、共享供体借料与中性核查。"""
from pathlib import Path
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import design_forward as fwd
import design_arc_sections as sec
import design_west_arc_report as base
from mythic.tectonics.crust import Crust
from mythic.figures.style import setup_fonts
from mythic.figures.netcdf import open_netcdf
from design_west_arc_c5_report import hashes

OUT = ROOT / "figures/design/v2/forward/u35_c9"
CACHE = ROOT / ".cache/u35_c9"
CASES = {"baseline": "无B5减薄基准", "progressive": "35→0Ma持续减薄，总β1.5"}


def save(fig, name, note):
    fig.text(.01, .012, "U35-C9 · " + fwd.code_version() +
        "\n历史/伸展处方 user_override；通量观察 procedural_constrained；查询 interpolated；L7约60km\n" + note,
        fontsize=8, va="bottom")
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def neutrality(case):
    old = ROOT / ".cache/u35_c8" / case
    now = CACHE / case
    arrays = base.compare_npz(now / "forward.npz", old / "forward.npz")
    oldmeta, meta = base.read_json(old / "forward.json"), base.read_json(now / "forward.json")
    with (old / "final_state.pkl").open("rb") as f:
        a = pickle.load(f)["engine"]
    with (now / "final_state.pkl").open("rb") as f:
        b = pickle.load(f)["engine"]
    result = {"saved_arrays": arrays, "budget_identical": meta["budget"] == oldmeta["budget"],
        "events_identical": meta["events"] == oldmeta["events"],
        "native_fields_identical": all(np.array_equal(getattr(a.c, k), getattr(b.c, k), equal_nan=True) for k in Crust.FIELDS),
        "alive_identical": np.array_equal(a.alive, b.alive),
        "rotations_identical": set(a.R) == set(b.R) and all(np.array_equal(a.R[k], b.R[k]) for k in a.R)}
    with open_netcdf(old / "world.nc") as a, open_netcdf(now / "world.nc") as b:
        result["world_arrays_identical"] = set(a.data_vars) == set(b.data_vars) and all(np.array_equal(a[k].values, b[k].values, equal_nan=True) for k in a.data_vars)
    if not arrays["identical"] or not all(v for k, v in result.items() if k != "saved_arrays"):
        raise RuntimeError("只读观测改变了原结果")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    result = {"checkpoint": "U35-C9", "code_base": "705239f", "method_class": "procedural_constrained",
        "physical_change": False, "cases": {}, "protected_sha256": hashes()}
    c8 = base.read_json(ROOT / "data/worlds/design_v2/u35_c8_diagnostic.json")
    result["protected_unchanged_from_C8"] = result["protected_sha256"] == c8["protected_sha256"]
    cases = {}
    for case in CASES:
        d = base.read_json(CACHE / case / "coupling.json")
        ledger = base.read_json(CACHE / case / "ledger.json")
        flow = [r for r in d["flow_edges_since_36Ma"] if r["target_uid"] == 52985 and r["t_Ma"] < 36]
        borrow = [r for r in d["borrow_transfers_since_36Ma"] if r["donor_uid"] == 52985 and r["t_Ma"] < 36]
        ci = ledger["watch_uids"].index(52985)
        area = c8["center_area_km2"]
        t = np.arange(35, -1, -1)
        incoming = np.array([sum(max(0, r["volume_into_target_km3"]) for r in flow if r["t_Ma"] == tt) / area for tt in t])
        outgoing = np.array([sum(min(0, r["volume_into_target_km3"]) for r in flow if r["t_Ma"] == tt) / area for tt in t])
        donors = {}
        for r in flow:
            entry = donors.setdefault(str(r["neighbor_uid"]), {"origin": r["neighbor_origin"], "plate": r["neighbor_plate"],
                "in_km3": 0., "out_km3": 0., "C_frame_neighbor_lon_lat": r["C_frame_neighbor_lon_lat"]})
            v = r["volume_into_target_km3"]
            entry["in_km3" if v > 0 else "out_km3"] += abs(v)
        end, start = ledger["timeline"][-1], next(r for r in ledger["timeline"] if r["t_Ma"] == 36)
        expected = {name: (end["process_cumulative_km3"][name][ci] - start["process_cumulative_km3"][name][ci]) for name in ("lateral_flow", "commit_birth_or_borrow")}
        checks = d["all_commit_borrow_checks"]
        center_events = list({r["t_Ma"]: {k: r[k] for k in ("t_Ma", "donor_T_before_km", "donor_T_after_all_holes_km", "donor_available_km3", "donor_total_taken_km3")} for r in borrow}.values())
        proof = {"flow_sum_vs_stock_error_km3": float(sum(r["volume_into_target_km3"] for r in flow) - expected["lateral_flow"]),
            "borrow_sum_vs_stock_error_km3": float(-sum(r["volume_to_hole_km3"] for r in borrow) - expected["commit_birth_or_borrow"])}
        if max(abs(v) for v in proof.values()) > 1e-6:
            raise RuntimeError("中心逐边/逐洞记录与原收支不一致")
        row = {"neutrality_vs_C8": neutrality(case), "attribution_closure": proof,
            "flow_reconstruction_max_error_km3": d["flow_reconstruction_max_error_km3"],
            "flow_global_balance_max_km3": d["flow_global_balance_max_km3"],
            "borrow_reconstruction_max_error_km3": d["borrow_reconstruction_max_error_km3"],
            "center_flow_in_km": float(incoming.sum()), "center_flow_out_km": float(outgoing.sum()),
            "center_net_flow_km": float((incoming + outgoing).sum()), "center_neighbor_sources": donors,
            "center_borrow_equivalent_km": float(-sum(r["volume_to_hole_km3"] for r in borrow) / area),
            "center_borrow_events": center_events,
            "center_max_overdraft_equivalent_km": max([max(0., r["donor_total_taken_km3"] - r["donor_available_km3"]) / area for r in borrow], default=0.),
            "global_borrow_checks": checks,
            "global_overdraft_km3_sum": sum(r["overdraft_km3"] for r in checks),
            "global_overdraft_max_equivalent_km": max(r["max_overdraft_equivalent_km"] for r in checks),
            "standard_figures_reused": "figures/stage1/design_v2_forward_v2_0Ma_u35_c8_progressive/index.html" if case == "progressive" else "基准与C8基准相同，全球形态参考C8已有对照；基准没有独立标准图组",
            "A": c8["candidates"][case]["A"], "b_verdict": c8["candidates"].get(case, {}).get("b_verdict")}
        result["cases"][case] = row
        cases[case] = dict(data=d, ledger=ledger, flow=flow, borrow=borrow, times=t, incoming=incoming, outgoing=outgoing)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, (case, title) in zip(axes, CASES.items()):
        d = cases[case]
        ax.bar(d["times"], d["incoming"], color="#aa7146", label="邻居输入")
        ax.bar(d["times"], d["outgoing"], color="#5787ac", label="输往邻居")
        ax.plot(d["times"], np.cumsum(d["incoming"] + d["outgoing"]), c="#704187", label="累计净输入")
        ax.set(title=title + "：初始UID52985真实逐边流动", ylabel="体积÷中心面积 km", xlim=(36, -1))
        ax.grid(alpha=.2)
        ax.legend(ncol=3)
    axes[-1].set_xlabel("Ma；单步输入/输出与累计净量同图，均不是现存厚度")
    fig.tight_layout(rect=(0, .13, 1, .97))
    save(fig, "57_B5_flow_edges.png", "两次完整演化与C8各自352数组、预算、事件和最终原生状态相同；侧向流动是厚度扩散代理。")

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, (case, title) in zip(axes, CASES.items()):
        flow = cases[case]["flow"]
        for uid, row in result["cases"][case]["center_neighbor_sources"].items():
            segment = [r for r in flow if str(r["neighbor_uid"]) == uid]
            a = np.mean([r["C_frame_neighbor_lon_lat"] for r in segment], axis=0)
            b = np.mean([r["C_frame_target_lon_lat"] for r in segment], axis=0)
            net = (row["in_km3"] - row["out_km3"]) / c8["center_area_km2"]
            ax.plot([a[0], b[0]], [a[1], b[1]], c="#aa7146" if net > 0 else "#5787ac", lw=1 + abs(net))
            ax.scatter(*a, s=40, c="#888")
            ax.annotate(f"UID{uid}\n净{net:+.2f}km", a, xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.scatter(*b, marker="*", s=160, c="#e1b638", edgecolors="#333", label="UID52985平均位置")
        ax.set(title=title, xlabel="洲C现今参照系经度 °E", ylabel="纬度 °N", aspect=1.)
        ax.margins(x=.2, y=.17)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout(rect=(0, .13, 1, .94))
    save(fig, "58_B5_flow_sources.png", "线宽为35→0Ma累计净转移的等效厚度；节点为材料宿主平均位置，连线不是断层或真实流线。")

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for ax, (case, title) in zip(axes, CASES.items()):
        rows = result["cases"][case]["center_borrow_events"]
        for r in rows:
            ax.plot([r["t_Ma"], r["t_Ma"]], [r["donor_T_before_km"], r["donor_T_after_all_holes_km"]], c="#945c42", marker="o")
        ax.axhline(34, c="#6c4586", ls="--", label="名义借料下限34km")
        ax.set(title=title + "：同一次提交全部洞的合计扣款", ylabel="中心柱厚度 km", xlim=(36, -1))
        ax.grid(alpha=.2)
        ax.legend()
    axes[-1].set_xlabel("提交年代 Ma；竖线只表示借料前后，不是整个时间步")
    fig.tight_layout(rect=(0, .13, 1, .97))
    save(fig, "59_B5_shared_donor.png", "多个洞分别按相同前态分配，合并扣款可能超过供体可用体积；实际观测与库存变化独立核对。")

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for case, title in CASES.items():
        checks = result["cases"][case]["global_borrow_checks"]
        t = [r["t_Ma"] for r in checks]
        axes[0].plot(t, [r["overdraft_km3"] / 1e6 for r in checks], label=title)
        axes[1].plot(t, [r["max_overdraft_equivalent_km"] for r in checks], label=title)
    axes[0].set(ylabel="共享余额超额分配 百万km³", title="完整300→0Ma的每次提交：超额分配不是全球不守恒量")
    axes[1].set(ylabel="最严重供体超扣等效 km", xlabel="Ma", xlim=(300, 0))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout(rect=(0, .13, 1, .97))
    save(fig, "60_shared_budget_global.png", "旧规则仍将扣款体积交给新洞，总量可守恒；缺陷是多个洞重复占用同一份34km以上的供体余额。")
    result["limitations"] = ["未改变物理结果，不能称为新谷地改善", "累计超额分配是跨提交重复统计，不能当作现存体积或新增体积",
        "名义34km下限不是地球所有岩石圈的物理边界", "本轮没有解释全部碰撞输入的现实成因"]
    base.write_json(ROOT / "data/worlds/design_v2/u35_c9_diagnostic.json", sec.clean(result))
    html = "<!doctype html><meta charset='utf-8'><title>U35-C9 B5补厚与共享借料</title><style>body{font-family:system-ui;max-width:1500px;margin:24px auto;background:#eee}img{max-width:100%}</style><h1>U35-C9：实际通量与共享供体余额</h1><p>只读观测，未改变世界。原始全时段账本在缓存中；下列图保留输入、输出和借料前后，不能只凭账本关闭B5。</p>"
    for name in ("57_B5_flow_edges.png", "58_B5_flow_sources.png", "59_B5_shared_donor.png", "60_shared_budget_global.png"):
        html += f"<h2>{name}</h2><img src='{name}'>"
    html += "<p><a href='../u35_c8/index.html'>C8基准/候选地貌与标准图组</a></p>"
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({k: {n: r[n] for n in ("center_flow_in_km", "center_flow_out_km", "center_max_overdraft_equivalent_km", "global_overdraft_max_equivalent_km")} for k, r in result["cases"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
