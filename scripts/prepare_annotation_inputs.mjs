import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

function usage() {
  return "Usage: node scripts/prepare_annotation_inputs.mjs <benchmark.xlsx> <output-directory>";
}

function rowsToObjects(rows) {
  const [headers, ...dataRows] = rows;
  return dataRows.map((row) => Object.fromEntries(headers.map((header, index) => [header, row[index] ?? null])));
}

function excelSerialToIso(value) {
  if (typeof value !== "number") {
    throw new Error(`Expected Excel date serial, received ${JSON.stringify(value)}`);
  }
  const epochMilliseconds = Math.round((value - 25_569) * 86_400_000);
  return new Date(epochMilliseconds).toISOString();
}

function sha256(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function toJsonLines(records) {
  return `${records.map((record) => JSON.stringify(record)).join("\n")}\n`;
}

const [workbookArgument, outputArgument] = process.argv.slice(2);
if (!workbookArgument || !outputArgument) {
  throw new Error(usage());
}

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const workbookPath = path.resolve(workbookArgument);
const outputDirectory = path.resolve(outputArgument);
const universePath = path.join(repositoryRoot, "config", "target_universe.json");
const schemaPath = path.join(repositoryRoot, "config", "news_feature_schema.json");

const [workbookBytes, universeText, schemaText] = await Promise.all([
  fs.readFile(workbookPath),
  fs.readFile(universePath, "utf8"),
  fs.readFile(schemaPath, "utf8"),
]);
const universe = JSON.parse(universeText);
const schema = JSON.parse(schemaText);

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const annotations = rowsToObjects(workbook.worksheets.getItem("Annotation Benchmark").getUsedRange(true).values);
const metadata = rowsToObjects(workbook.worksheets.getItem("Sample Metadata").getUsedRange(true).values);
const metadataById = new Map(metadata.map((row) => [row.article_id, row]));

if (annotations.length !== 300) {
  throw new Error(`Expected 300 benchmark articles, found ${annotations.length}`);
}

const inputs = annotations.map((article, index) => {
  const meta = metadataById.get(article.article_id);
  if (!meta) {
    throw new Error(`Missing metadata for ${article.article_id}`);
  }
  const targetTicker = meta.focus_ticker || universe.default_target_for_sector_and_macro_candidates;
  const target = universe.targets[targetTicker];
  if (!target) {
    throw new Error(`Unknown target ticker ${targetTicker} for ${article.article_id}`);
  }
  return {
    row_number: index + 1,
    article_id: article.article_id,
    time_published_utc: excelSerialToIso(article.time_published_utc),
    source: article.source,
    headline: article.title,
    article_text: article.summary,
    vendor_tickers: article.vendor_tickers
      ? String(article.vendor_tickers).split(",").map((item) => item.trim()).filter(Boolean)
      : [],
    target: {
      company: target.company,
      ticker: targetTicker,
      sector: universe.sector,
      sector_benchmark: universe.sector_benchmark,
      known_sector_peers: target.peers,
    },
  };
});

const ids = new Set(inputs.map((record) => record.article_id));
if (ids.size !== inputs.length) {
  throw new Error("Duplicate article_id values detected in benchmark input");
}

await fs.mkdir(outputDirectory, { recursive: true });
await fs.writeFile(path.join(outputDirectory, "all_inputs.jsonl"), toJsonLines(inputs), "utf8");

const batchSize = 100;
for (let start = 0; start < inputs.length; start += batchSize) {
  const batch = inputs.slice(start, start + batchSize);
  const first = String(start + 1).padStart(3, "0");
  const last = String(start + batch.length).padStart(3, "0");
  await fs.writeFile(path.join(outputDirectory, `batch_${first}_${last}.jsonl`), toJsonLines(batch), "utf8");
}

const targetCounts = Object.fromEntries(Object.keys(universe.targets).map((ticker) => [ticker, 0]));
for (const record of inputs) {
  targetCounts[record.target.ticker] = (targetCounts[record.target.ticker] ?? 0) + 1;
}

const manifest = {
  generated_at_utc: new Date().toISOString(),
  record_count: inputs.length,
  schema_name: schema.schema_name,
  schema_version: schema.schema_version,
  source_workbook: path.basename(workbookPath),
  source_workbook_sha256: sha256(workbookBytes),
  schema_sha256: sha256(Buffer.from(schemaText, "utf8")),
  target_universe_sha256: sha256(Buffer.from(universeText, "utf8")),
  default_target_for_missing_focus: universe.default_target_for_sector_and_macro_candidates,
  target_counts: targetCounts,
  excluded_from_annotator_input: [
    "sampling_stratum_not_ground_truth",
    "vendor sentiment scores and labels",
    "query source",
    "quarter",
  ],
};
await fs.writeFile(path.join(outputDirectory, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

process.stdout.write(`Prepared ${inputs.length} blinded inputs in ${outputDirectory}\n`);
