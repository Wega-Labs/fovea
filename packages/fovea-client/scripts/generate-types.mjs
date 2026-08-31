import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const schemaPath = resolve(packageRoot, "../../schema/fovea-protocol-v1.json");
const outputPath = resolve(packageRoot, "src/generated/protocol.ts");
const schema = JSON.parse(readFileSync(schemaPath, "utf8"));
const definitions = schema.$defs;

if (!definitions || typeof definitions !== "object") {
  throw new Error("protocol schema does not contain an object-valued $defs");
}

function pascalCase(value) {
  return value
    .split("_")
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join("");
}

function definitionName(name) {
  if (name.startsWith("command_")) {
    return `${pascalCase(name.slice("command_".length))}Command`;
  }
  return pascalCase(name);
}

function propertyName(name) {
  return /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name) ? name : JSON.stringify(name);
}

function renderType(node, indent = 0) {
  if (Object.hasOwn(node, "const")) {
    return JSON.stringify(node.const);
  }
  if (Array.isArray(node.enum)) {
    return node.enum.map((value) => JSON.stringify(value)).join(" | ");
  }
  if (Array.isArray(node.anyOf)) {
    return node.anyOf.map((choice) => renderType(choice, indent)).join(" | ");
  }
  if (node.type === "array") {
    return `ReadonlyArray<${renderType(node.items, indent)}>`;
  }
  if (node.type === "object") {
    const required = new Set(node.required ?? []);
    const lines = ["{"];
    for (const [name, property] of Object.entries(node.properties ?? {})) {
      const optional = required.has(name) ? "" : "?";
      const type = renderType(property, indent + 2);
      lines.push(
        `${" ".repeat(indent + 2)}readonly ${propertyName(name)}${optional}: ${type};`,
      );
    }
    lines.push(`${" ".repeat(indent)}}`);
    return lines.join("\n");
  }
  if (node.type === "string") return "string";
  if (node.type === "number" || node.type === "integer") return "number";
  if (node.type === "boolean") return "boolean";
  if (node.type === "null") return "null";
  throw new Error(`unsupported schema node: ${JSON.stringify(node)}`);
}

const eventDefinitions = [];
const commandDefinitions = [];
const lines = [
  "// Generated from schema/fovea-protocol-v1.json. Do not edit by hand.",
  "",
];

for (const [schemaName, definition] of Object.entries(definitions)) {
  if (definition.type !== "object") {
    throw new Error(`definition ${schemaName} is not an object`);
  }
  const name = definitionName(schemaName);
  const required = new Set(definition.required ?? []);
  lines.push(`export interface ${name} {`);
  for (const [field, property] of Object.entries(definition.properties ?? {})) {
    const optional = required.has(field) ? "" : "?";
    lines.push(
      `  readonly ${propertyName(field)}${optional}: ${renderType(property, 2)};`,
    );
  }
  lines.push("}", "");
  if (schemaName.startsWith("command_")) {
    commandDefinitions.push([name, definition.properties?.cmd?.const]);
  } else {
    eventDefinitions.push([name, definition.properties?.type?.const]);
  }
}

lines.push("export type FoveaEvent =");
for (const [name] of eventDefinitions) lines.push(`  | ${name}`);
lines[lines.length - 1] += ";";
lines.push("", "export type FoveaCommand =");
for (const [name] of commandDefinitions) lines.push(`  | ${name}`);
lines[lines.length - 1] += ";";

lines.push("", "export const FOVEA_EVENT_TYPES = [");
for (const [, discriminator] of eventDefinitions) {
  if (typeof discriminator !== "string") throw new Error("event is missing type const");
  lines.push(`  ${JSON.stringify(discriminator)},`);
}
lines.push("] as const;", "", "export const FOVEA_COMMAND_TYPES = [");
for (const [, discriminator] of commandDefinitions) {
  if (typeof discriminator !== "string") throw new Error("command is missing cmd const");
  lines.push(`  ${JSON.stringify(discriminator)},`);
}
lines.push("] as const;");

const generated = `${lines.join("\n")}\n`;
if (process.argv.includes("--check")) {
  const committed = readFileSync(outputPath, "utf8");
  if (committed !== generated) {
    console.error("generated protocol types are stale; run npm run generate");
    process.exitCode = 1;
  }
} else {
  writeFileSync(outputPath, generated);
}
