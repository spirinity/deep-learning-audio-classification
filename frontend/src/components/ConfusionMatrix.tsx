import type { ConfusionResponse } from "@/lib/types";
import type { Copy } from "@/lib/i18n";
import { methodColor } from "@/lib/constants";
import { Card, CardContent } from "@/components/ui/card";

interface Props {
  data: ConfusionResponse;
  copy: Copy;
}

const SHORT_LABELS: Record<string, string> = {
  Animals: "Animals",
  "Natural soundscapes": "Natural",
  "Human non-speech": "Human",
  "Interior/domestic": "Interior",
  "Exterior/urban": "Exterior",
};

function short(category: string): string {
  return SHORT_LABELS[category] ?? category;
}

// Solid pastel mint shading by normalized value (no gradients), per the
// Cursor design language. Diagonal-heavy = errors stay within the category.
function cellStyle(value: number): React.CSSProperties {
  return {
    backgroundColor: `color-mix(in srgb, #9fc9a2 ${Math.round(value * 100)}%, var(--card))`,
  };
}

export default function ConfusionMatrix({ data, copy }: Props) {
  if (!data.available || data.methods.length === 0) {
    return <p className="text-sm text-muted-foreground">{copy.confusionEmpty}</p>;
  }

  const { categories } = data;

  return (
    <div className="flex flex-col gap-6">
      <p className="text-sm text-muted-foreground">{copy.confusionDescription}</p>
      <div className="grid gap-6 lg:grid-cols-3">
        {data.methods.map((m) => (
          <Card key={m.method}>
            <CardContent className="py-4">
              <div className="mb-3 flex items-center gap-2">
                <span
                  className="size-2.5 rounded-full"
                  style={{ backgroundColor: methodColor(m.method) }}
                />
                <span className="text-sm font-medium text-foreground">
                  {m.method}
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="border-separate border-spacing-0.5 text-[11px]">
                  <thead>
                    <tr>
                      <th className="p-1" />
                      <th
                        className="p-1 text-[10px] font-normal tracking-[0.06em] text-muted-foreground uppercase"
                        colSpan={categories.length}
                      >
                        {copy.confusionPredicted}
                      </th>
                    </tr>
                    <tr>
                      <th className="p-1" />
                      {categories.map((c) => (
                        <th
                          key={c}
                          className="p-1 text-center font-medium text-muted-foreground"
                          title={c}
                        >
                          {short(c)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {categories.map((rowCat, i) => (
                      <tr key={rowCat}>
                        <th
                          className="p-1 text-right font-medium text-muted-foreground"
                          title={rowCat}
                        >
                          {short(rowCat)}
                        </th>
                        {categories.map((colCat, j) => {
                          const count = m.matrix[i][j];
                          const norm = m.normalized[i][j];
                          return (
                            <td
                              key={colCat}
                              className="rounded-sm p-1 text-center tabular-nums text-foreground"
                              style={cellStyle(norm)}
                              title={`${rowCat} → ${colCat}: ${count} (${(norm * 100).toFixed(1)}%)`}
                            >
                              {count}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="mt-2 text-[10px] text-muted-foreground">
                {copy.confusionTrue} ↓ / {copy.confusionPredicted} →
              </p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
