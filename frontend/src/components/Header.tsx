import { Badge } from "@/components/ui/badge";

export default function Header() {
  return (
    <header className="flex flex-col items-center text-center">
      <Badge
        variant="secondary"
        className="mb-5 text-[11px] font-semibold tracking-[0.08em] text-muted-foreground uppercase"
      >
        INTERSPEECH 2023 · LAION-CLAP · ESC-50
      </Badge>
      <h1 className="display text-4xl text-foreground sm:text-6xl">
        Sound Classifier Comparison
      </h1>
      <p className="mt-4 max-w-2xl text-base text-muted-foreground sm:text-lg">
        Compare{" "}
        <span className="text-foreground">Zero-Shot CLAP</span>,{" "}
        <span className="text-foreground">Proto-LC</span>, and{" "}
        <span className="text-foreground">Supervised MLP</span> on the
        50-class ESC-50 sound dataset.
      </p>
    </header>
  );
}
