import type { MethodInfo } from "@/lib/types";
import { INK, methodColor } from "@/lib/constants";
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Props {
  methods: MethodInfo[];
}

export default function MethodExplanation({ methods }: Props) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {methods.map((m) => {
        const color = methodColor(m.name);
        return (
          <Card key={m.name} size="sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span
                  className="size-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: color }}
                />
                {m.name}
              </CardTitle>
              <CardAction>
                {m.available ? (
                  <span
                    className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-[0.06em] uppercase"
                    style={{ backgroundColor: color, color: INK }}
                  >
                    Ready
                  </span>
                ) : (
                  <Badge variant="secondary" className="text-[10px]">
                    Unavailable
                  </Badge>
                )}
              </CardAction>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {m.description}
              </p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
