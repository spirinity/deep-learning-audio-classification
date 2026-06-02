// UI constants. Colors follow the Cursor DESIGN.md pastel "timeline" palette
// (peach / mint / blue / lavender / gold). They are used as pill BACKGROUNDS
// with warm-ink text — never as text color on the cream canvas.

export const INK = "#26251e";

// Cursor pastel categorical palette, one per ESC-50 category.
export const CATEGORY_COLORS: Record<string, string> = {
  Animals: "#dfa88f", // peach
  "Natural soundscapes": "#9fc9a2", // mint
  "Human non-speech": "#c08532", // gold
  "Interior/domestic": "#9fbbe0", // blue
  "Exterior/urban": "#c0a8dd", // lavender
  Unknown: "#e6e5e0", // surface-strong
};

export function categoryColor(category: string): string {
  return CATEGORY_COLORS[category] ?? CATEGORY_COLORS.Unknown;
}

// Stable display order + pastel per method (matches the comparison chart).
export const METHOD_ORDER = [
  "Zero-Shot CLAP",
  "Proto-LC",
  "Logistic Regression",
] as const;

export const METHOD_COLORS: Record<string, string> = {
  "Zero-Shot CLAP": "#9fbbe0", // blue
  "Proto-LC": "#c0a8dd", // lavender
  "Logistic Regression": "#9fc9a2", // mint
};

export function methodColor(method: string): string {
  return METHOD_COLORS[method] ?? "#9fbbe0";
}

export const UPLOAD_LIMITS = {
  maxFileSizeMb: 10,
  acceptExtensions: [".wav", ".mp3"],
  acceptMime: "audio/wav,audio/x-wav,audio/mpeg,.wav,.mp3",
};
