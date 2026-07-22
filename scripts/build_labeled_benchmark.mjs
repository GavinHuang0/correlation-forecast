import fs from "node:fs/promises";
import path from "node:path";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

function usage() {
  return "Usage: node scripts/build_labeled_benchmark.mjs <source.xlsx> <inputs.jsonl> <reference.jsonl> <validation-report.json> <output.xlsx>";
}

function parseJsonLines(text) {
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function excelColumnName(columnNumber) {
  let value = columnNumber;
  let name = "";
  while (value > 0) {
    value -= 1;
    name = String.fromCharCode(65 + (value % 26)) + name;
    value = Math.floor(value / 26);
  }
  return name;
}

const [sourceArgument, inputsArgument, referenceArgument, reportArgument, outputArgument] = process.argv.slice(2);
if (!sourceArgument || !inputsArgument || !referenceArgument || !reportArgument || !outputArgument) {
  throw new Error(usage());
}

const sourcePath = path.resolve(sourceArgument);
const inputsPath = path.resolve(inputsArgument);
const referencePath = path.resolve(referenceArgument);
const reportPath = path.resolve(reportArgument);
const outputPath = path.resolve(outputArgument);

const [inputsText, referenceText, reportText] = await Promise.all([
  fs.readFile(inputsPath, "utf8"),
  fs.readFile(referencePath, "utf8"),
  fs.readFile(reportPath, "utf8"),
]);
const inputs = parseJsonLines(inputsText);
const references = parseJsonLines(referenceText);
const report = JSON.parse(reportText);
const inputById = new Map(inputs.map((record) => [record.article_id, record]));

if (inputs.length !== 300 || references.length !== 300) {
  throw new Error(`Expected 300 inputs and references, received ${inputs.length} and ${references.length}`);
}
if (report.error_count !== 0) {
  throw new Error(`Cannot build labeled workbook with ${report.error_count} validation errors`);
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));
const referenceSheet = workbook.worksheets.add("ChatGPT Reference");
const summarySheet = workbook.worksheets.add("Reference Summary");
const protocolSheet = workbook.worksheets.add("Protocol v0.1");

const headers = [
  "row_number",
  "article_id",
  "time_published_utc",
  "source",
  "headline",
  "target_ticker",
  "relevance",
  "event_scope",
  "event_type",
  "affected_breadth",
  "target_direction",
  "sector_direction",
  "peer_effect",
  "explicit_surprise",
  "information_status",
  "transmission_channels",
  "affected_companies",
  "affected_sectors",
  "evidence_scope",
  "evidence_direction",
  "evidence_surprise",
  "abstain_reason",
  "quality_flags",
  "protocol_version",
];

const rows = references
  .slice()
  .sort((left, right) => left.row_number - right.row_number)
  .map((reference) => {
    const input = inputById.get(reference.article_id);
    if (!input) throw new Error(`Missing input ${reference.article_id}`);
    const labels = reference.labels;
    return [
      reference.row_number,
      reference.article_id,
      new Date(input.time_published_utc),
      input.source,
      input.headline,
      reference.target_ticker,
      labels.relevance,
      labels.event_scope,
      labels.event_type,
      labels.affected_breadth,
      labels.target_direction,
      labels.sector_direction,
      labels.peer_effect,
      labels.explicit_surprise,
      labels.information_status,
      labels.transmission_channels.join("; "),
      labels.affected_companies.join("; "),
      labels.affected_sectors.join("; "),
      labels.evidence.scope,
      labels.evidence.direction,
      labels.evidence.surprise,
      labels.abstain_reason ?? "",
      reference.quality_flags.join("; "),
      reference.protocol_version,
    ];
  });

const lastColumn = excelColumnName(headers.length);
referenceSheet.getRange(`A1:${lastColumn}${rows.length + 1}`).values = [headers, ...rows];
referenceSheet.showGridLines = false;
referenceSheet.freezePanes.freezeRows(1);
referenceSheet.freezePanes.freezeColumns(6);
referenceSheet.getRange(`A1:${lastColumn}1`).format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
};
referenceSheet.getRange(`A2:${lastColumn}${rows.length + 1}`).format = {
  font: { color: "#1F2937" },
  verticalAlignment: "top",
};
referenceSheet.getRange(`C2:C${rows.length + 1}`).format.numberFormat = "yyyy-mm-dd hh:mm";
referenceSheet.getRange(`E2:E${rows.length + 1}`).format.wrapText = true;
referenceSheet.getRange(`P2:W${rows.length + 1}`).format.wrapText = true;
referenceSheet.getRange(`A1:${lastColumn}${rows.length + 1}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#E5E7EB" },
  bottom: { style: "thin", color: "#CBD5E1" },
};
referenceSheet.getRange(`A1:A${rows.length + 1}`).format.columnWidth = 10;
referenceSheet.getRange(`B1:B${rows.length + 1}`).format.columnWidth = 24;
referenceSheet.getRange(`C1:C${rows.length + 1}`).format.columnWidth = 20;
referenceSheet.getRange(`D1:D${rows.length + 1}`).format.columnWidth = 18;
referenceSheet.getRange(`E1:E${rows.length + 1}`).format.columnWidth = 56;
referenceSheet.getRange(`F1:F${rows.length + 1}`).format.columnWidth = 13;
referenceSheet.getRange(`G1:O${rows.length + 1}`).format.columnWidth = 22;
referenceSheet.getRange(`P1:R${rows.length + 1}`).format.columnWidth = 28;
referenceSheet.getRange(`S1:U${rows.length + 1}`).format.columnWidth = 38;
referenceSheet.getRange(`V1:W${rows.length + 1}`).format.columnWidth = 28;
referenceSheet.getRange(`X1:X${rows.length + 1}`).format.columnWidth = 16;
referenceSheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 34;
referenceSheet.getRange(`A2:${lastColumn}${rows.length + 1}`).format.rowHeight = 42;
const referenceTable = referenceSheet.tables.add(`A1:${lastColumn}${rows.length + 1}`, true, "ChatGPTReferenceTable");
referenceTable.style = "TableStyleMedium2";

referenceSheet.getRange(`G2:G${rows.length + 1}`).conditionalFormats.add("containsText", {
  text: "irrelevant",
  format: { fill: "#F3F4F6", font: { color: "#6B7280" } },
});
referenceSheet.getRange(`G2:G${rows.length + 1}`).conditionalFormats.add("containsText", {
  text: "direct_target",
  format: { fill: "#DCFCE7", font: { color: "#166534" } },
});

summarySheet.showGridLines = false;
summarySheet.mergeCells("A1:F1");
summarySheet.getRange("A1").values = [["GPT-5.6 Sol Silver-Reference Annotation Summary"]];
summarySheet.getRange("A1:F1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
summarySheet.getRange("A1:F1").format.rowHeight = 34;
summarySheet.getRange("A3:B6").values = [
  ["Metric", "Value"],
  ["Reference annotations", null],
  ["Validation errors", report.error_count],
  ["Validation warnings", report.warning_count],
];
summarySheet.getRange("B4").formulas = [["=COUNTA('ChatGPT Reference'!B2:B301)"]];
summarySheet.getRange("A3:B3").format = {
  fill: "#DCE6F1",
  font: { bold: true, color: "#17365D" },
};
summarySheet.getRange("A3:B6").format.borders = { preset: "outside", style: "thin", color: "#94A3B8" };
summarySheet.getRange("A8:F8").merge();
summarySheet.getRange("A8").values = [[
  "Interpretation: these are GPT-5.6 Sol silver labels for measuring model agreement, not unquestioned ground truth. Evidence strings were checked against the supplied headline and summary.",
]];
summarySheet.getRange("A8:F8").format = {
  fill: "#FFF7ED",
  font: { color: "#9A3412", italic: true },
  wrapText: true,
};
summarySheet.getRange("A8:F8").format.rowHeight = 48;

const scopeValues = ["firm_specific", "peer_specific", "sector_wide", "macro_market", "mixed", "unclear"];
summarySheet.getRange("A11:B17").values = [["Event scope", "Count"], ...scopeValues.map((value) => [value, null])];
for (let row = 12; row <= 17; row += 1) {
  summarySheet.getRange(`B${row}`).formulas = [[`=COUNTIF('ChatGPT Reference'!H$2:H$301,A${row})`]];
}
summarySheet.getRange("A11:B11").format = {
  fill: "#DCE6F1",
  font: { bold: true, color: "#17365D" },
};
summarySheet.getRange("A11:B17").format.borders = {
  insideHorizontal: { style: "thin", color: "#E5E7EB" },
  top: { style: "thin", color: "#94A3B8" },
  bottom: { style: "thin", color: "#94A3B8" },
  left: { style: "thin", color: "#94A3B8" },
  right: { style: "thin", color: "#94A3B8" },
};
const scopeChart = summarySheet.charts.add("bar", summarySheet.getRange("A11:B17"));
scopeChart.title = "Reference Event-Scope Distribution";
scopeChart.hasLegend = false;
scopeChart.setPosition("D10", "K24");

summarySheet.getRange("A20:F23").values = [
  ["Reproducibility item", "Recorded value", null, null, null, null],
  ["Protocol version", references[0].protocol_version, null, null, null, null],
  ["Reference annotator", references[0].annotator, null, null, null, null],
  ["Join key", "article_id", null, null, null, null],
];
summarySheet.getRange("A20:B20").format = {
  fill: "#DCE6F1",
  font: { bold: true, color: "#17365D" },
};
summarySheet.getRange("A1:K24").format.font.name = "Aptos";
summarySheet.getRange("A1:A24").format.columnWidth = 28;
summarySheet.getRange("B1:B24").format.columnWidth = 22;
summarySheet.freezePanes.freezeRows(1);

const protocolRows = [
  ["Field", "Allowed values / rule"],
  ["relevance", "direct_target; sector_or_peer; macro_relevant; irrelevant; insufficient"],
  ["event_scope", "firm_specific; peer_specific; sector_wide; macro_market; mixed; unclear"],
  ["event_type", "earnings; guidance; product_technology; demand_customer_contract; supply_chain_capacity; regulation_trade_policy; analyst_action; corporate_action; legal_governance_operations; macro_market; other; unclear"],
  ["affected_breadth", "single_firm; several_same_sector; cross_sector; broad_market; unclear"],
  ["target_direction / sector_direction", "positive; negative; neutral; mixed; unknown; not_applicable"],
  ["peer_effect", "same_direction; opposite_direction; mixed; none_stated; unknown; not_applicable"],
  ["explicit_surprise", "positive; negative; mixed; none; unknown; requires explicit benchmark language"],
  ["information_status", "confirmed; scheduled_or_expected; rumor_or_unconfirmed; analysis_or_opinion; unclear"],
  ["transmission_channels", "At most two fixed-taxonomy channels"],
  ["affected entities", "Only explicitly named or explicitly covered companies/sectors"],
  ["evidence", "Short exact substring of supplied headline or summary"],
  ["abstain_reason", "Required explanation when supplied text is insufficient; otherwise null"],
  ["research restriction", "Semantic extraction only; never forecast return, volatility, beta, correlation, or profitability"],
];
protocolSheet.getRange(`A1:B${protocolRows.length}`).values = protocolRows;
protocolSheet.showGridLines = false;
protocolSheet.freezePanes.freezeRows(1);
protocolSheet.getRange("A1:B1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
};
protocolSheet.getRange(`A2:B${protocolRows.length}`).format.wrapText = true;
protocolSheet.getRange(`A1:B${protocolRows.length}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#E5E7EB" },
  top: { style: "thin", color: "#94A3B8" },
  bottom: { style: "thin", color: "#94A3B8" },
  left: { style: "thin", color: "#94A3B8" },
  right: { style: "thin", color: "#94A3B8" },
};
protocolSheet.getRange(`A1:A${protocolRows.length}`).format.columnWidth = 34;
protocolSheet.getRange(`B1:B${protocolRows.length}`).format.columnWidth = 92;
protocolSheet.getRange(`A2:B${protocolRows.length}`).format.rowHeight = 42;
const protocolTable = protocolSheet.tables.add(`A1:B${protocolRows.length}`, true, "ProtocolTable");
protocolTable.style = "TableStyleMedium2";

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

const preview = await workbook.render({
  sheetName: "Reference Summary",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(`${outputPath}.summary.png`, new Uint8Array(await preview.arrayBuffer()));

const inspection = await workbook.inspect({
  kind: "sheet,table,formula,computedStyle",
  sheetId: "ChatGPT Reference",
  range: "A1:X6",
  maxChars: 12_000,
  tableMaxRows: 6,
  tableMaxCols: 24,
});
await fs.writeFile(`${outputPath}.inspect.ndjson`, `${inspection.ndjson}\n`, "utf8");
process.stdout.write(`Built ${outputPath}\n`);
