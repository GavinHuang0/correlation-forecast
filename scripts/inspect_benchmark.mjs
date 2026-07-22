import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = process.argv[2];
const previewDirectory = process.argv[3];
if (!workbookPath) {
  throw new Error("Usage: node scripts/inspect_benchmark.mjs <workbook.xlsx>");
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 8_000,
  tableMaxRows: 4,
  tableMaxCols: 24,
  tableMaxCellChars: 120,
});
process.stdout.write(`${summary.ndjson}\n`);

for (const sheetName of ["Annotation Benchmark", "Sample Metadata"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange(true);
  const rows = used.values;
  process.stdout.write(`\n${sheetName}: ${rows.length} rows x ${rows[0]?.length ?? 0} columns\n`);
  process.stdout.write(`${JSON.stringify(rows.slice(0, 3), null, 2)}\n`);
}

if (previewDirectory) {
  await fs.mkdir(previewDirectory, { recursive: true });
  const preview = await workbook.render({
    sheetName: "README",
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    `${previewDirectory}/source_workbook_readme.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
