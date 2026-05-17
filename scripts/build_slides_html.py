"""Build reports/slides.html - a self-contained, flight-ticket themed deck.

The whole project is told as a journey: the question we boarded with, the
choices we made, the turbulence we hit and how we adjusted course, and where
we landed. Natural narrative voice; real weather-active numbers; a few real
charts embedded as base64 so the single .html file works anywhere offline.
"""
from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "slides.html"


def b64(name: str) -> str:
    p = FIGS / name
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


PLANE = ('<svg class="pl" viewBox="0 0 24 24" aria-hidden="true">'
         '<path d="M2 16l9-3 4-10 2 0-2 9 7-2 0 2-7 2 2 7-2 0-4-7-9 3z" '
         'fill="currentColor"/></svg>')


def route(active: int) -> str:
    dots = "".join(
        f'<span class="dot {"on" if i < active else ""}"></span>'
        f'{"" if i == 9 else "<span class=seg></span>"}'
        for i in range(10))
    return (f'<div class="route"><span class="ap">QST</span>'
            f'<div class="line">{dots}<span class="jet" '
            f'style="left:calc({active}/10*100% - 10px)">{PLANE}</span></div>'
            f'<span class="ap">ANS</span></div>')


def slide(n, kicker, html):
    return (f'<section class="slide" id="s{n}"><div class="deck">'
            f'<div class="top"><span class="kick">{kicker}</span>'
            f'{route(n)}<span class="pg">{n:02d}<i>/10</i></span></div>'
            f'<div class="stage">{html}</div></div></section>')


def tag(t, kind="d"):
    return f'<span class="tag {kind}">{t}</span>'


def turb(struggle, fix):
    return ('<div class="turb"><div class="th">&#9888; TURBULENCE - '
            'HOW WE ADJUSTED COURSE</div>'
            f'<p class="ts">{struggle}</p>'
            f'<p class="tf">&#10148; {fix}</p></div>')


def stat(big, small):
    return f'<div class="stat"><b>{big}</b><span>{small}</span></div>'


# ---- the journey -----------------------------------------------------------
S = []

S.append('<section class="slide cover" id="s0"><div class="deck">'
  '<div class="bp">'
    '<div class="bp-main">'
      '<div class="bp-air">EC&#8202;261 AIRWAYS<span>· data-science journey ·</span></div>'
      '<h1>Profitable Flight-Delay Prediction</h1>'
      '<div class="bp-route"><div><b>QUESTION</b><span>can a cheap ticket pay '
      'for itself?</span></div>' + PLANE +
      '<div><b>ANSWER</b><span>a diagnosed negative result</span></div></div>'
      '<div class="bp-grid">'
        '<div><b>PASSENGERS</b><span>Habib · Issam · Adam · Lama · Salma · Sanad</span></div>'
        '<div><b>FLIGHT</b><span>ML-2026</span></div>'
        '<div><b>OPERATOR</b><span>IE University</span></div>'
      '</div>'
    '</div>'
    '<div class="bp-stub">'
      '<div class="hole"></div>'
      '<div class="st-air">EC&#8202;261</div>'
      '<div class="st-f"><b>SEAT</b><span>10</span></div>'
      '<div class="st-f"><b>DURATION</b><span>20 MIN</span></div>'
      '<div class="bar"></div>'
    '</div>'
  '</div>'
  '<p class="cover-sub">Can ML find cheap flights whose EC261 payout beats the '
  'ticket? This is the story of how we tested it - and what the data told us.</p>'
  '</div></section>')

S.append(slide(1, "PRE-DEPARTURE · WHY WE BOARDED",
  '<div class="cols"><div class="col">'
  '<p class="lead">Our journey started with a quietly tempting idea.</p>'
  '<p>EU Regulation 261 pays a passenger <b>&euro;250&ndash;600</b> when a '
  'flight lands 3&nbsp;hours+ late - but only if the airline itself is to '
  'blame. Weather, air-traffic control and strikes are exempt.</p>'
  '<p>So we asked: if we could spot the <i>cheap</i> tickets most likely to '
  'earn that payout, would buying them be worth it?</p>'
  + tag("CHOICE · we model a DECISION, not accuracy", "d") +
  '<p>From day one we framed it as money, not metrics: <b>buy a ticket only if '
  'expected payout &gt; price + claim + travel cost</b>. The real question '
  'became - is that signal even learnable?</p>'
  '</div><div class="col mid">'
  '<div class="fcard"><div class="fc-h">THE BET</div>'
  '<div class="bet"><span>Cheap ticket</span>'+PLANE+'<span>Carrier delay '
  '&ge;3h?</span>'+PLANE+'<span class="pay">&euro;250&ndash;600</span></div>'
  '<p class="fc-n">A journey to find out if the asymmetry is real.</p></div>'
  '</div></div>'))

S.append(slide(2, "BOARDING · LOADING THE CARGO",
  '<div class="cols"><div class="col">'
  '<p class="lead">Every journey needs the right cargo - real flights, '
  'not toy data.</p>'
  '<p>We loaded <b>6,965,247</b> real US flights from 2024 (BTS on-time + '
  'cause-of-delay), and kept <b>3.89&nbsp;M</b> European flights aside for a '
  'second opinion later.</p>'
  + tag("CHOICE · the label that defines the project", "d") +
  '<p>A flight only counts if it is 3h+ late <i>and</i> the airline caused it. '
  'That single decision moved the base rate from <b>1.43% &rarr; 1.18%</b> - '
  'modelling raw delay would have chased payouts that never arrive.</p>'
  + turb("6.97&nbsp;M rows were far too heavy to tune a fleet of models on a "
         "laptop - the plane couldn't take off.",
         "we re-routed: a seeded, month&times;label-stratified <b>150k</b> "
         "sample that preserves the exact 1.18% mix, fully reproducible.") +
  '</div><div class="col mid">'
  + stat("6.97 M", "real 2024 US flights loaded")
  + stat("150 k", "seeded stratified sample flown")
  + stat("1.18%", "EC261-eligible base rate (kept)")
  + stat("3.89 M", "EU flights reserved for later") +
  '</div></div>'))

S.append(slide(3, "PRE-FLIGHT CHECKS · EDA & PREP",
  '<div class="cols"><div class="col">'
  '<p class="lead">Before take-off, we ran the instruments over the data.</p>'
  '<p>Only ~1% of flights pay out, so &ldquo;accuracy&rdquo; is a vanity gauge '
  '(always-skip already scores 98.8%). We chose to fly by '
  '<b>PR-AUC, calibration and money ROI</b> instead.</p>'
  + tag("CHOICE · keep the long-haul outliers", "d") +
  '<p>EC261 pays on a <i>step function</i> of distance, so clipping long '
  'flights would have quietly falsified the reward. We kept them.</p>'
  + turb("the weather feed came back completely empty - a gauge reading zero.",
         "we traced a disconnected loader, re-wired it, and weather now "
         "informs <b>88.7%</b> of flights.") +
  '</div><div class="col mid">'
  '<img class="chart" src="IMG_CORR" alt="feature correlation heatmap">'
  '<p class="cap">An early instrument reading: features barely correlate with '
  'the payout - a quiet hint of what was ahead.</p>'
  '</div></div>'))

S.append(slide(4, "THE FLIGHT PLAN · PIPELINE",
  '<p class="lead center">We sealed the whole route into one pipeline so '
  'training could never peek at the future.</p>'
  '<div class="flow">'
  + "".join(f'<div class="wp"><span>{t}</span></div>' +
            ("" if i==6 else f'<i class="conn">{PLANE}</i>')
            for i,t in enumerate(["Raw BTS<br>6.97 M","Seeded<br>150 k",
            "Booking<br>features","Past-only<br>rolling stats",
            "Column<br>transformer","Model fleet<br>+ calibration",
            "&tau;* profit<br>decision"])) +
  '</div>'
  '<div class="cols2">'
  '<div>'+tag("CHOICE · leakage-safe by construction","d")+
  '<p>14-day booking horizon, strictly-past rolling stats, time-ordered '
  'splits, and forbidden columns dropped at the door - defence in depth.</p></div>'
  + '<div>' + turb("parts of the notebook had never been flown end-to-end.",
         "a full dry-run surfaced <b>three</b> breaking issues; we fixed every "
         "one before departure.") + '</div>'
  '</div>'))

S.append(slide(5, "CHOOSING THE AIRCRAFT · MODELS",
  '<div class="cols"><div class="col">'
  '<p class="lead">We didn&rsquo;t fly just one plane.</p>'
  '<p>A whole fleet: from a trivial baseline up through logistic regression, '
  'trees, <b>Random Forest, XGBoost</b> and a neural net - so we could see '
  'exactly how much signal each could squeeze out.</p>'
  + tag("CHOICE · calibrate everything","d")+
  '<p>Expected-value maths only works with honest odds, so every model was '
  '<b>isotonic-calibrated</b>.</p>'
  + tag("CHOICE · optimise money, not F1","d")+
  '<p>We derived a per-flight break-even threshold <b>&tau;*(price, '
  'distance)</b> in closed form and tuned a custom profit score directly.</p>'
  '</div><div class="col mid">'
  + stat("6 models", "trivial &rarr; advanced fleet")
  + stat("isotonic", "calibration on every model")
  + stat("&tau;*(T,d)", "per-flight money threshold")
  + stat("Grid/Rand/Bayes", "tuning matched to each model") +
  '</div></div>'))

S.append(slide(6, "READING THE INSTRUMENTS · EVALUATION",
  '<div class="cols"><div class="col">'
  '<p class="lead">How do you judge a money model? Carefully.</p>'
  '<p>We compared on <b>both validation and test</b>, with cross-validated '
  'error bars (reference PR-AUC 0.031&nbsp;&plusmn;&nbsp;0.014) and the full '
  'set of ROC, learning and confusion curves.</p>'
  + turb("one model&rsquo;s gauge overflowed - numeric instability mid-curve.",
         "we found the cause and swapped to a stable instrument, rather than "
         "tape over the warning light.") +
  '<p>The biggest lesson: <b>calibration</b> was the line between a model that '
  'gambles and one that wisely abstains.</p>'
  '</div><div class="col mid">'
  '<img class="chart" src="IMG_CAL" alt="calibration before vs after">'
  '<p class="cap">Before vs after calibration - the difference between betting '
  'and abstaining.</p>'
  '</div></div>'))

S.append(slide(7, "WHAT THE INSTRUMENTS SHOWED",
  '<div class="cols"><div class="col">'
  '<p class="lead">Then came the twist.</p>'
  '<p>Every calibrated model, independently, made the <b>same</b> call: '
  '<b>do not buy</b>. ROI 0%, zero tickets, every time.</p>'
  '<p>At first it looked like a broken gauge. So we did the arithmetic by '
  'hand - and found it wasn&rsquo;t the model at all.</p>'
  + tag("THE DISCOVERY · it is structural","x")+
  '<p>To break even you need ~<b>63%</b> confidence a flight will be '
  'compensated. The most delay-prone route&times;airline only reaches '
  '~<b>7.7%</b>; the average is <b>1.2%</b>. An <b>~8&times;</b> gap no model '
  'can close.</p>'
  '</div><div class="col mid">'
  '<div class="fcard"><div class="fc-h">THE BREAK-EVEN GAP</div>'
  '<div class="bar-row"><span>Need to break even</span>'
  '<div class="bk"><i style="width:100%;background:var(--coral)"></i></div><b>63%</b></div>'
  '<div class="bar-row"><span>Best cohort ever</span>'
  '<div class="bk"><i style="width:12%;background:var(--amber)"></i></div><b>7.7%</b></div>'
  '<div class="bar-row"><span>Average flight</span>'
  '<div class="bk"><i style="width:2%;background:var(--mint)"></i></div><b>1.2%</b></div>'
  '<p class="fc-n">Not a model failure - the economics simply don&rsquo;t close.</p>'
  '</div></div></div>'))

S.append(slide(8, "A SECOND OPINION · EUROPEAN SKIES",
  '<div class="cols"><div class="col">'
  '<p class="lead">We took the same model over a different sky.</p>'
  '<p>Flown across <b>3.89&nbsp;M real European flights</b>, it didn&rsquo;t '
  'just fail to help - its ranking <b>inverted</b> (the more confident it was, '
  'the <i>less</i> likely a delay; Spearman &rho;&nbsp;=&nbsp;&minus;1.00).</p>'
  '<p>An independent confirmation, from data it had never seen, that the '
  'signal isn&rsquo;t there to be caught.</p>'
  + tag("REASSURANCE · the leakage guard held","d")+
  '<p>Interpretation showed schedule &amp; route features carry the (weak) '
  'signal, while forbidden columns contributed exactly <b>0.000</b>.</p>'
  '</div><div class="col mid">'
  '<img class="chart" src="IMG_DEC" alt="EU decile inversion">'
  '<img class="chart sm" src="IMG_SHAP" alt="SHAP importance">'
  '</div></div>'))

S.append(slide(9, "LANDING · WHAT WE LEARNED",
  '<div class="cols"><div class="col">'
  '<p class="lead">Every honest journey ends with a clear-eyed debrief.</p>'
  '<p>Limits we own: fares were modelled (real ones are paywalled), it is one '
  'year of data, and Europe&rsquo;s schedules are recorded differently. But '
  'the conclusion rests on <b>arithmetic</b>, not a single model.</p>'
  + tag("NEXT TIME · how we&rsquo;d fly it again","d")+
  '<p>Real fare data, and a two-stage label that separates &ldquo;will it be '
  'delayed&rdquo; from &ldquo;will the airline be liable&rdquo;.</p>'
  '<p class="land">ML correctly shows the EC261 ticket-buying strategy is '
  '<b>structurally unprofitable</b> - proven from two independent skies.</p>'
  '</div><div class="col mid">'
  '<div class="fcard end"><div class="fc-h">ARRIVED</div>'
  '<p class="thanks">Thank you for flying with us.</p>'
  '<div class="crew">Habib &middot; Issam &middot; Adam &middot; Lama '
  '&middot; Salma &middot; Sanad</div>'+PLANE+'</div>'
  '</div></div>'))

CSS = """
:root{--navy:#0a2540;--ocean:#15507a;--sky:#cfe6f4;--sky2:#eaf4fb;
--cream:#fdf8ee;--ink:#16323f;--coral:#ef6f53;--amber:#f0a83c;
--mint:#2f9e8f;--mut:#6c8190;--line:#dfe6ea}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{font-family:"Segoe UI",-apple-system,"Helvetica Neue",Arial,sans-serif;
color:var(--ink);background:var(--navy);
scroll-snap-type:y mandatory;overflow-y:scroll;scroll-behavior:smooth}
.slide{height:100vh;display:flex;align-items:center;justify-content:center;
scroll-snap-align:start;padding:3vh 3vw;
background:linear-gradient(180deg,var(--sky) 0%,var(--sky2) 55%,#fff 100%)}
.deck{width:min(96vw,1180px);aspect-ratio:16/9;background:#fff;
border-radius:22px;box-shadow:0 24px 60px rgba(10,37,64,.28);
padding:40px 48px;display:flex;flex-direction:column;position:relative;
overflow:hidden}
.deck:before{content:"";position:absolute;inset:0;
background:radial-gradient(1200px 300px at 90% -8%,rgba(58,143,183,.10),transparent 60%);
pointer-events:none}
.pl{width:1em;height:1em;display:inline-block;vertical-align:-.15em;
color:var(--ocean)}
.top{display:flex;align-items:center;gap:18px;margin-bottom:8px}
.kick{font:700 12px/1 "Segoe UI";letter-spacing:.16em;color:var(--amber);
background:var(--navy);padding:9px 14px;border-radius:999px;white-space:nowrap}
.route{flex:1;display:flex;align-items:center;gap:10px;color:var(--mut)}
.route .ap{font:700 11px/1 ui-monospace,monospace;letter-spacing:.12em}
.line{flex:1;position:relative;display:flex;align-items:center;height:18px}
.dot{width:7px;height:7px;border-radius:50%;background:#cfd9df}
.dot.on{background:var(--ocean)}
.seg{flex:1;height:2px;background:repeating-linear-gradient(90deg,#cfd9df 0 5px,transparent 5px 10px)}
.jet{position:absolute;font-size:18px;color:var(--ocean)}
.pg{font:800 16px/1 ui-monospace,monospace;color:var(--navy)}
.pg i{color:var(--mut);font-style:normal;font-weight:600;font-size:12px}
.stage{flex:1;display:flex;flex-direction:column;justify-content:center;
padding-top:6px}
.cols{display:grid;grid-template-columns:1.15fr .85fr;gap:34px;align-items:center}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:18px}
.col p{font-size:16px;line-height:1.55;margin:9px 0;color:#2b4250}
.lead{font-size:21px!important;font-weight:700;color:var(--navy)!important;
line-height:1.35!important;margin:0 0 6px!important}
.lead.center{text-align:center;margin-bottom:14px!important}
b{color:var(--navy)}
.tag{display:inline-block;font:700 12px/1 "Segoe UI";letter-spacing:.04em;
color:#fff;background:var(--ocean);padding:8px 13px;border-radius:8px;
margin:12px 0 4px;position:relative}
.tag:before{content:"";position:absolute;left:9px;top:-5px;width:6px;height:6px;
border-radius:50%;background:#fff;box-shadow:0 0 0 2px var(--ocean)}
.tag.x{background:var(--coral)}.tag.x:before{box-shadow:0 0 0 2px var(--coral)}
.turb{background:#fff4ec;border:1px solid #f3c4ad;border-left:5px solid var(--coral);
border-radius:12px;padding:13px 16px;margin:14px 0}
.turb .th{font:800 11px/1 "Segoe UI";letter-spacing:.08em;color:var(--coral);
margin-bottom:7px}
.turb .ts{font-size:14.5px;color:#4a3b34;margin:2px 0}
.turb .tf{font-size:14.5px;color:var(--mint);font-weight:700;margin:6px 0 0}
.mid{display:flex;flex-direction:column;gap:12px}
.stat{background:linear-gradient(135deg,var(--navy),var(--ocean));color:#fff;
border-radius:14px;padding:14px 18px}
.stat b{display:block;color:#fff;font-size:26px;font-weight:800;letter-spacing:.01em}
.stat span{font-size:13px;color:#cfe0ec}
.fcard{background:var(--cream);border:1px dashed #d9c9a6;border-radius:16px;
padding:18px 20px;text-align:center}
.fc-h{font:800 11px/1 ui-monospace,monospace;letter-spacing:.18em;
color:var(--ocean);margin-bottom:14px}
.fc-n{font-size:13px;color:var(--mut);margin-top:14px;font-style:italic}
.bet{display:flex;align-items:center;justify-content:center;gap:12px;
flex-wrap:wrap;font-weight:700;color:var(--navy);font-size:15px}
.bet .pay{color:var(--coral)}
.bet .pl{color:var(--mut)}
.bp-route{display:flex;align-items:center;justify-content:space-between;
gap:18px;margin:18px 0;padding:14px 0;border-top:1px dashed #d9c9a6;
border-bottom:1px dashed #d9c9a6}
.bp-route .pl{font-size:30px;color:var(--ocean)}
.bp-route b{display:block;font:800 11px/1 ui-monospace,monospace;
letter-spacing:.14em;color:var(--ocean)}
.bp-route span{font-size:14px;color:#5b6f7b}
.cap{font-size:12.5px;color:var(--mut);font-style:italic;margin-top:10px;
text-align:center}
.chart{width:100%;border:1px solid var(--line);border-radius:12px;
background:#fff;box-shadow:0 6px 16px rgba(10,37,64,.10)}
.chart.sm{margin-top:10px}
.flow{display:flex;align-items:center;justify-content:center;flex-wrap:nowrap;
gap:0;margin:6px 0 4px}
.wp{background:linear-gradient(135deg,var(--ocean),var(--sky));color:#fff;
border-radius:12px;padding:14px 8px;width:118px;text-align:center;
font:700 12px/1.3 "Segoe UI";box-shadow:0 6px 14px rgba(10,37,64,.16)}
.wp:nth-child(5){background:linear-gradient(135deg,#3a6f8a,#86b6cf)}
.conn{color:#9fb3bf;font-size:16px;padding:0 6px}
.center{text-align:center}
.land{margin-top:14px!important;font-size:17px!important;font-weight:700;
color:var(--navy)!important;border-top:2px solid var(--line);padding-top:12px}
.bar-row{display:flex;align-items:center;gap:12px;margin:11px 0;
font-size:13.5px;color:#42586a}
.bar-row span{width:140px;text-align:right}
.bk{flex:1;height:16px;background:#eef3f5;border-radius:8px;overflow:hidden}
.bk i{display:block;height:100%;border-radius:8px}
.bar-row b{width:42px;color:var(--navy)}
.end{display:flex;flex-direction:column;align-items:center;gap:8px}
.thanks{font-size:20px;font-weight:800;color:var(--navy)}
.crew{font-size:14px;color:var(--mut);letter-spacing:.03em}
.end .pl{font-size:34px;color:var(--ocean);margin-top:6px}
/* cover */
.cover{background:linear-gradient(160deg,var(--navy),var(--ocean) 70%,#2f87b0)}
.cover .deck{background:transparent;box-shadow:none;align-items:center;
justify-content:center}
.bp{display:flex;background:var(--cream);border-radius:22px;overflow:hidden;
box-shadow:0 30px 70px rgba(0,0,0,.4);max-width:920px;width:100%}
.bp-main{flex:1;padding:38px 44px}
.bp-air{font:800 13px/1 ui-monospace,monospace;letter-spacing:.2em;
color:var(--ocean)}
.bp-air span{display:block;font-weight:600;letter-spacing:.12em;color:var(--mut);
margin-top:6px;font-size:11px}
.bp h1{font-size:34px;color:var(--navy);margin:16px 0 4px;line-height:1.15}
.bp-grid{display:flex;gap:34px;flex-wrap:wrap}
.bp-grid b,.st-f b{display:block;font:800 10px/1 ui-monospace,monospace;
letter-spacing:.14em;color:var(--ocean)}
.bp-grid span{font-size:13.5px;color:#465c68}
.bp-stub{width:210px;background:#f3ead4;border-left:3px dashed #cdbb91;
padding:30px 24px;position:relative;display:flex;flex-direction:column;gap:16px}
.hole{position:absolute;left:-14px;top:50%;width:28px;height:28px;
border-radius:50%;background:var(--navy);transform:translateY(-50%)}
.st-air{font:800 14px/1 ui-monospace,monospace;letter-spacing:.18em;
color:var(--navy)}
.st-f span{font-size:15px;color:var(--navy);font-weight:700}
.bar{margin-top:auto;height:54px;background:repeating-linear-gradient(90deg,
var(--navy) 0 3px,transparent 3px 6px,var(--navy) 6px 8px,transparent 8px 13px)}
.cover-sub{color:#dCEAF4;text-align:center;margin-top:26px;font-size:16px;
max-width:760px;line-height:1.5}
nav{position:fixed;right:20px;bottom:18px;display:flex;gap:8px;z-index:50}
nav a{width:40px;height:40px;border-radius:50%;background:rgba(255,255,255,.92);
color:var(--navy);display:flex;align-items:center;justify-content:center;
font-size:20px;text-decoration:none;box-shadow:0 6px 16px rgba(0,0,0,.25);
font-weight:800}
@media print{
 body{overflow:visible}
 .slide{height:auto;min-height:100vh;page-break-after:always;break-after:page}
 nav{display:none}
 @page{size:1280px 720px;margin:0}
}
"""

HTML = (
 "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
 "<meta name='viewport' content='width=device-width,initial-scale=1'>"
 "<title>EC261 Airways - a data-science journey</title>"
 f"<style>{CSS}</style></head><body>"
 + "".join(S)
 + "<nav><a href='#s0' title='restart'>&#8635;</a>"
   "<a href='#s9' title='end'>&#8594;</a></nav>"
 + "<script>"
   "let i=0,S=[...document.querySelectorAll('.slide')];"
   "addEventListener('keydown',e=>{"
   "if(e.key==='ArrowRight'||e.key==='ArrowDown'||e.key===' '){i=Math.min(i+1,S.length-1);S[i].scrollIntoView()}"
   "if(e.key==='ArrowLeft'||e.key==='ArrowUp'){i=Math.max(i-1,0);S[i].scrollIntoView()}});"
   "new IntersectionObserver(es=>es.forEach(x=>{if(x.isIntersecting)i=S.indexOf(x.target)}),"
   "{threshold:.5}).observe?S.forEach(s=>new IntersectionObserver(es=>es.forEach(x=>{"
   "if(x.isIntersecting)i=S.indexOf(x.target)}),{threshold:.5}).observe(s)):0;"
   "</script></body></html>"
)

HTML = (HTML
  .replace("IMG_CORR", b64("01_correlations.png"))
  .replace("IMG_CAL", b64("05_calibration_before_after.png"))
  .replace("IMG_DEC", b64("06_decile_monotonicity.png"))
  .replace("IMG_SHAP", b64("05_kernel_shap.png")))

OUT.write_text(HTML, encoding="utf-8")
kb = len(HTML) // 1024
print(f"[slides-html] wrote {OUT} ({kb} KB, {len(S)} slides, self-contained)")
