"""Assembles results/gallery.html from results/figures/ + the summary tables.
Run after plot_results.py, from anywhere (paths are relative to this file)."""

import base64
from pathlib import Path

import pandas as pd

FIG_DIR = Path(__file__).parent / "figures"
FIG_NAMES = [
    "fig1_reach_success_rate.png", "fig1_push_success_rate.png",
    "fig2_reach_returns.png", "fig2_push_returns.png",
    "fig3_reach_train_success.png", "fig3_push_train_success.png",
    "fig4_reach_entropy_coef.png", "fig4_push_entropy_coef.png",
    "fig5_reach_goal_distance.png", "fig5_push_goal_distance.png",
    "fig6_push_ablation.png",
]
FIGS = {name: base64.b64encode((FIG_DIR / name).read_bytes()).decode("ascii")
        for name in FIG_NAMES}

df = pd.read_csv(Path(__file__).parent / "figures/summary.csv")
abl = pd.read_csv(Path(__file__).parent / "figures/ablation_summary.csv")

COND_LABEL = {"dense": "dense", "sparse": "sparse", "dense_her": "dense + HER", "sparse_her": "sparse + HER"}
COND_ORDER = ["dense", "sparse", "dense_her", "sparse_her"]

def summary_rows(task):
    sub = df[df.task == task]
    rows = []
    for cond in COND_ORDER:
        c = sub[sub.condition == cond]
        if c.empty:
            continue
        n = len(c)
        final = c.final_success.mean()
        reached = (c.steps_to_90pct.notna()).sum()
        med = c.steps_to_90pct.median()
        med_s = f"{med:,.0f}" if pd.notna(med) else "—"
        auc = c.auc.mean() if "auc" in c.columns else c.success_auc.mean()
        rows.append(f"""
        <tr>
          <td>{COND_LABEL[cond]}</td>
          <td class="num">{final*100:.1f}%</td>
          <td class="num">{reached}/{n}</td>
          <td class="num">{med_s}</td>
          <td class="num">{auc:.3f}</td>
        </tr>""")
    return "\n".join(rows)

def ablation_rows():
    rows = []
    for nsg in sorted(abl.n_sampled_goal.unique()):
        c = abl[abl.n_sampled_goal == nsg]
        n = len(c)
        final = c.final_success.mean()
        reached = (c.steps_to_90pct.notna()).sum()
        med = c.steps_to_90pct.median()
        med_s = f"{med:,.0f}" if pd.notna(med) else "—"
        auc = c.success_auc.mean()
        rows.append(f"""
        <tr>
          <td>n_sampled_goal = {int(nsg)}</td>
          <td class="num">{final*100:.1f}%</td>
          <td class="num">{reached}/{n}</td>
          <td class="num">{med_s}</td>
          <td class="num">{auc:.3f}</td>
        </tr>""")
    return "\n".join(rows)

REACH_TABLE = summary_rows("reach")
PUSH_TABLE = summary_rows("push")
ABLATION_TABLE = ablation_rows()

def img(name, alt):
    return f'<img src="data:image/png;base64,{FIGS[name]}" alt="{alt}" loading="lazy">'

HTML = f"""<title>SAC-HER Results — FetchReach &amp; FetchPush</title>
<style>
  :root {{
    --bg: #F6F7F9;
    --surface: #FFFFFF;
    --surface-2: #EEF1F4;
    --border: #DEE3E9;
    --fg: #14181F;
    --fg-muted: #5C6572;
    --accent: #B3541E;
    --accent-soft: #FBEEE3;
    --good: #1B8F63;
    --bad: #B3311C;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #101317;
      --surface: #171B21;
      --surface-2: #1D222A;
      --border: #2B313B;
      --fg: #E9ECF1;
      --fg-muted: #9BA4B0;
      --accent: #E29352;
      --accent-soft: #2B2016;
      --good: #3FCB93;
      --bad: #E2685A;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #101317; --surface: #171B21; --surface-2: #1D222A; --border: #2B313B;
    --fg: #E9ECF1; --fg-muted: #9BA4B0; --accent: #E29352; --accent-soft: #2B2016;
    --good: #3FCB93; --bad: #E2685A;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--fg); margin: 0; padding: 2.5rem 1.25rem 4rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, Arial, sans-serif;
    line-height: 1.6;
  }}
  .mono {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace; font-variant-numeric: tabular-nums; }}
  .ar {{ direction: rtl; text-align: right; unicode-bidi: isolate; }}
  .page {{ max-width: 980px; margin: 0 auto; display: flex; flex-direction: column; gap: 2.5rem; }}
  header {{ display: flex; flex-direction: column; gap: 0.5rem; }}
  .eyebrow {{ color: var(--accent); font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; }}
  h1 {{ font-size: 1.6rem; margin: 0; text-wrap: balance; }}
  .headline {{
    background: var(--accent-soft); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.1rem 1.3rem; font-size: 1rem;
  }}
  .headline b {{ color: var(--accent); }}
  section {{ display: flex; flex-direction: column; gap: 1rem; }}
  h2 {{ font-size: 1.05rem; margin: 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }}
  .fig-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1rem; }}
  figure {{ margin: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem; }}
  figure img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ margin-top: 0.7rem; font-size: 0.85rem; color: var(--fg-muted); }}
  figcaption b {{ color: var(--fg); }}
  table {{ border-collapse: collapse; width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; font-size: 0.88rem; }}
  th, td {{ padding: 0.55rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: var(--surface-2); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--fg-muted); }}
  td.num {{ font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
  tr:last-child td {{ border-bottom: none; }}
  .table-wrap {{ overflow-x: auto; border-radius: 10px; }}
  .note {{ color: var(--fg-muted); font-size: 0.85rem; }}
  footer {{ border-top: 1px solid var(--border); padding-top: 1rem; font-size: 0.78rem; color: var(--fg-muted); }}
</style>

<div class="page">

  <header>
    <div class="eyebrow mono">SAC-HER · results</div>
    <h1 class="ar">نتائج الدراسة الكاملة — 70 تجربة تدريب</h1>
  </header>

  <div class="headline ar">
    على المهمة السهلة (FetchReach) كل الحالات نجحت تقريباً بنفس المستوى. على المهمة الصعبة (FetchPush)،
    <b>SAC بدون HER توقف عند 5.4% نجاح (صفر من 10 محاولات وصلت 90%)</b>، بينما
    <b>SAC مع HER وصل 90.7% نجاح (9 من 10 محاولات)</b> — بنفس إشارة المكافأة الفقيرة بالضبط.
    HER صارت الفرق الحرفي بين "تعلّم" و"ما تعلّم إطلاقاً".
  </div>

  <section>
    <h2 class="ar">1) النتيجة الأساسية — منحنى النجاح</h2>
    <div class="fig-grid">
      <figure>
        {img("fig1_reach_success_rate.png", "FetchReach success rate curves")}
        <figcaption class="ar"><b>FetchReach (سهلة):</b> الأربع حالات كلها بتوصل لنجاح شبه كامل. الفرق بينهم بسرعة الوصول بس — dense وdense+HER وsparse+HER كلهم متقاربين (~6-8 آلاف خطوة)، sparse لحاله أبطأ بشكل واضح (~22 ألف خطوة).</figcaption>
      </figure>
      <figure>
        {img("fig1_push_success_rate.png", "FetchPush success rate curves")}
        <figcaption class="ar"><b>FetchPush (صعبة):</b> sparse لحاله (أحمر) عالق تحت 10% طول التدريب. sparse+HER (أخضر) بيصعد بثبات لـ~90%. هاد الرسم هو "الدليل" الأساسي على الفرضية.</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2 class="ar">2) جدول الأرقام الكامل</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>FetchReach — الحالة</th><th>نجاح نهائي</th><th>seeds وصلت 90%</th><th>وسيط خطوات لـ90%</th><th>AUC</th></tr></thead>
        <tbody>{REACH_TABLE}</tbody>
      </table>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>FetchPush — الحالة</th><th>نجاح نهائي</th><th>seeds وصلت 90%</th><th>وسيط خطوات لـ90%</th><th>AUC</th></tr></thead>
        <tbody>{PUSH_TABLE}</tbody>
      </table>
    </div>
    <p class="note ar">AUC = المساحة تحت منحنى النجاح (كفاءة العينة الكلية طول التدريب، مش بس بالنهاية). 10 seeds لكل حالة.</p>
  </section>

  <section>
    <h2 class="ar">3) مقاييس ثانوية — العائد وسرعة النجاح أثناء التدريب</h2>
    <div class="fig-grid">
      <figure>
        {img("fig2_push_returns.png", "FetchPush episode returns")}
        <figcaption class="ar">العائد لكل حلقة (episode return) على Push. الخط الرمادي المتقطع (-50) هو "فشل دائم". sparse+HER بتبتعد بوضوح عن هالخط، sparse بتضل عالقة قريبة منه.</figcaption>
      </figure>
      <figure>
        {img("fig3_push_train_success.png", "FetchPush training-time rolling success")}
        <figcaption class="ar">نسبة النجاح أثناء التدريب نفسه (rolling، مش تقييم منفصل) — نفس الاتجاه العام، إشارة إضافية إنو النتيجة مش مصادفة قياس.</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2 class="ar">4) الآلية — ليش HER بتنفع؟ (معامل الاستكشاف)</h2>
    <div class="fig-grid">
      <figure>
        {img("fig4_push_entropy_coef.png", "SAC entropy coefficient over training on Push")}
        <figcaption class="ar">
          معامل α هو "قديش الخوارزمية لسا عم تستكشف عشوائياً" — كل ما قلّ، صار الروبوت أكثر ثقة (وأقل استكشاف) بقراراته.
          <b>sparse لحاله (أحمر): المعامل بينهار لقريب الصفر</b> — يعني SAC "اقتنع" إنو لقى حل جيد وبيوقف الاستكشاف، رغم إنو فعلياً عالق عند 5% نجاح بس! هاي مشكلة كلاسيكية: ثقة زايدة بحل سيء بسبب غياب أي إشارة تصحح المسار.
          <b>sparse+HER (أخضر): المعامل بيضل أعلى بـ~10 أضعاف طول التدريب</b> — يعني الروبوت بيضل يستكشف بشكل صحي كفاية لين يلاقي الحل الحقيقي. هاي جزء من تفسير "ليش" HER بتنفع: مش بس بتعطي أهداف بديلة، كمان بتمنع الخوارزمية من الاقتناع المبكر بحل غلط.
        </figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2 class="ar">5) الآلية — توزيع بُعد الأهداف (المنهج التدريجي الضمني)</h2>
    <div class="fig-grid">
      <figure>
        {img("fig5_reach_goal_distance.png", "Goal distance distribution over training on Reach")}
        <figcaption class="ar">على Reach، الوسيط (الخط الغامق) بينزل بثبات من ~0.10 متر لـ~0.02 متر — دليل واضح على "منهج تدريجي": الأهداف يلي الروبوت عم يتدرب عليها فعلياً بتصير أقرب مع الوقت كل ما تحسّن أداؤه.</figcaption>
      </figure>
      <figure>
        {img("fig5_push_goal_distance.png", "Goal distance distribution over training on Push")}
        <figcaption class="ar">على Push، نفس المؤشر أقل وضوحاً — الوسيط قريب من الصفر طول التدريب (لأنو ~80% من العينات معاد تسميتها أصلاً بهالإعداد)، والانتشار (p10-p90) بيضل شبه ثابت. المنهج التدريجي موجود بس أصعب نشوفه بهالمقياس المجمّع — نقطة صادقة نذكرها بالتحليل، مش نتيجة سلبية.</figcaption>
      </figure>
    </div>
  </section>

  <section>
    <h2 class="ar">6) تجربة الـ ablation — قديش مرة نستخدم HER؟</h2>
    <div class="fig-grid">
      <figure>
        {img("fig6_push_ablation.png", "HER relabeling ratio ablation on Push")}
        <figcaption class="ar">مقارنة n_sampled_goal = 1، 4، 8 على Push. الإعداد الأعلى (8، أزرق فاتح) بيتعلم أسرع شوي بالبداية، بس بالنهاية الثلاثة بيوصلوا لمنطقة متقاربة (~80-90%). يعني الفايدة موجودة بس بتتشبّع، مش بتزيد بشكل خطي مع زيادة عدد إعادة التسمية.</figcaption>
      </figure>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>الإعداد</th><th>نجاح نهائي</th><th>seeds وصلت 90%</th><th>وسيط خطوات لـ90%</th><th>AUC</th></tr></thead>
        <tbody>{ABLATION_TABLE}</tbody>
      </table>
    </div>
    <p class="note ar">5 seeds لكل إعداد. n_sampled_goal=4 معاد استخدامها من التجربة الأساسية (نفس الـ5 محاولات الأولى من sparse+HER)، مش تجربة منفصلة.</p>
  </section>

  <footer class="ar">
    مبني من results/figures/ (plot_results.py) على كامل بيانات الـ70 تجربة. التفاصيل الكاملة وقرارات كل خطوة بـ EXPERIMENT_LOG.md.
  </footer>

</div>
"""

out_path = Path(__file__).parent / "gallery.html"
out_path.write_text(HTML)
print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")
