/* P1-2 复用测试：用通用 verdictCard + statEvidenceHtml 渲染"研究空白候选卡"
 * 数据来自真实 OpenAlex 探测（2026-11-23），断言结构与视觉要素。
 */
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'static', 'index.html'), 'utf8');
const start = html.indexOf('function verdictCard(s) {');
const end = html.indexOf('function vcCardHtml(v, paperTitle, quotes) {');
if (start < 0 || end < 0) throw new Error('未找到 verdictCard/statEvidenceHtml');
const code = html.slice(start, end);

const ESC = new Map([['&', '&amp;'], ['<', '&lt;'], ['>', '&gt;'], ['\u00a0', '&nbsp;']]);
global.document = {
  createElement: () => ({
    set textContent(v) { this._t = String(v == null ? '' : v); },
    get innerHTML() { return this._t.replace(/[&<>\u00a0]/g, c => ESC.get(c)); },
  }),
};
global.esc = s => { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; };
(0, eval)(code);

// P1-2 的判定枚举与配色（与 P0-2 的 supported/hallucination 完全分离）
const GAP_META = {
  red:    { label: '🔴 红海/新赛道', color: 'var(--red)',   bg: 'var(--red-lt)' },
  blue:   { label: '🟡 早期蓝海',   color: 'var(--amber)', bg: 'var(--amber-lt)' },
  gap:    { label: '🟢 空白候选',   color: 'var(--green)', bg: 'var(--green-lt)' },
  pseudo: { label: '⚫ 伪空白',     color: 'var(--muted)', bg: 'var(--bg)' },
};

const CASES = [
  { level: 'red', a: 'large language model', b: 'industrial control', nA: 255699, nB: 18094,
    nAB: 151, years: { 2023: 1, 2024: 16, 2025: 50, 2026: 83 },
    reason: '近三年占 99%，已进入爆发期，切入即红海' },
  { level: 'blue', a: 'spiking neural network', b: 'wireless communication', nA: 15469,
    nB: 126406, nAB: 36, years: { 2023: 1, 2024: 7, 2025: 12, 2026: 7 },
    reason: '总量小但逐年递增，处于早期布局窗口' },
  { level: 'pseudo', a: 'federated learning', b: 'deep sea observatory', nA: 62022,
    nB: 174, nAB: 0, years: {},
    reason: 'B 域仅 174 篇，零命中来自领域过窄而非无人研究' },
  { level: 'gap', a: 'graph neural network', b: 'power system state estimation', nA: 59996,
    nB: 1579, nAB: 19, years: { 2023: 2, 2024: 3, 2025: 3, 2026: 5 },
    reason: '稳定小规模且无爆发迹象，存在未覆盖的细化方向' },
];

let pass = 0, fail = 0;
const assert = (n, c) => { c ? (pass++, console.log('PASS', n)) : (fail++, console.log('FAIL', n)); };

const out = CASES.map(c => {
  const m = GAP_META[c.level];
  const card = verdictCard({
    color: m.color, bg: m.bg, badge: m.label,
    meta: [{ cls: 'veid', html: `${esc(c.a)} × ${esc(c.b)}` }, { cls: 'vpaper', html: '候选 #1' }],
    claim: `「${c.a} × ${c.b}」交叉方向覆盖度偏低，可作为选题候选。${c.reason}`,
    blocks: [
      `<div class="vlvl"><b>判定依据</b>：${c.reason}</div>`,
    ],
    evidence: statEvidenceHtml({
      formula: `title_and_abstract.search:"${c.a}" "${c.b}"`,
      nA: c.nA, nB: c.nB, nAB: c.nAB, years: c.years,
      note: `渗透率 ${(c.nAB / Math.min(c.nA, c.nB) * 100).toFixed(3)}%（双向取小）`,
    }),
  });
  return { c, card };
});

out.forEach(({ c, card }) => {
  const tag = `${c.level} ${c.a}×${c.b}`;
  assert(`${tag}: 使用通用卡片骨架`, card.includes('class="vcard"') && card.includes('class="vhead"'));
  assert(`${tag}: 判定徽章文案正确`, card.includes(GAP_META[c.level].label));
  assert(`${tag}: 含统计证据槽`, card.includes('class="vstat"') && card.includes('class="vst-tab"'));
  assert(`${tag}: 无 PDF 截图标签`, !card.includes('<img'));
  assert(`${tag}: 三项计数齐全`, card.includes(String(c.nAB)) && card.includes(String(c.nA)));
  assert(`${tag}: 覆盖 A×B 检索式`, card.includes('title_and_abstract.search'));
  const bars = (card.match(/class="vst-bar"/g) || []).length;
  assert(`${tag}: 年度柱数=${Object.keys(c.years).length}`, bars === Object.keys(c.years).length);
  // P0-2 的判定枚举绝不能泄漏进来
  assert(`${tag}: 未泄漏 P0-2 枚举`, !/核实通过|幻觉剔除|abstained/.test(card));
});

// 极值：全年份为 0 / 缺失 → 不崩且无柱
const zero = statEvidenceHtml({ nA: 1, nB: 2, nAB: 0, years: {} });
assert('空年分布不渲染柱', !zero.includes('vst-bar') && zero.includes('vstat'));

// 最大柱高度恒为 34px（归一化正确）
const one = statEvidenceHtml({ nA: 1, nB: 1, nAB: 1, years: { 2026: 1 } });
assert('单年柱高=34px', one.includes('height:34px'));

console.log(`\n--- ${pass} passed, ${fail} failed ---`);
fs.writeFileSync(path.join(__dirname, 'preview_gap_card.html'),
  `<meta charset="utf-8"><body style="font-family:sans-serif;background:#f4f6fb;padding:20px">
   <style>${html.slice(html.indexOf('/* ---- P0-2'), html.indexOf('/* ---- 检索前引导'))}</style>
   ${out.map(o => o.card).join('\n')}`, 'utf8');
process.exit(fail ? 1 : 0);
